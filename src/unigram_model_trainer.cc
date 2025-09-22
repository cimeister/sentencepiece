// Slimmed version of sentencepiece model trainer code

#include "unigram_model_trainer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>  // std::getenv, std::strtod
#include <cerrno>   // errno
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "filesystem.h"
#include "normalizer.h"
#include "pretokenizer_for_training.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/esaxx/esa.hxx"  // Suffix array library.
#include "trainer_interface.h"
#include "unicode_script.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {

constexpr char32 kSentenceBoundary = 0x0000;

double Digamma(double x) {
  double result = 0.0;
  for (; x < 7; ++x) result -= 1 / x;
  x -= 1.0 / 2.0;
  const double xx = 1.0 / x;
  const double xx2 = xx * xx;
  const double xx4 = xx2 * xx2;
  result += std::log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
            (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
  return result;
}

inline bool ParseFloat(const std::string& s, float* out) {
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || errno != 0 || !std::isfinite(v)) return false;
  *out = static_cast<float>(v);
  return true;
}

template <typename IT>
void ToLogProb(IT begin, IT end) {
  double sum = 0.0;
  for (auto it = begin; it != end; ++it) sum += static_cast<double>(it->second);
  if (sum <= 0.0) return;
  const double logsum = std::log(sum);
  for (auto it = begin; it != end; ++it) {
    it->second = static_cast<float>(std::log(static_cast<double>(it->second)) - logsum);
  }
}

// Simple bounded max-heap used by SP’s seeding code.
template <class T>
class BoundedPriorityQueue {
 public:
  explicit BoundedPriorityQueue(size_t size) : size_(size) {}
  void push(T elem, int64_t score) {
    if (queue_.size() > 4 * size_) resize_();
    if (sorted_ && queue_.size() >= size_ && queue_[size_ - 1].second > score) return;
    queue_.emplace_back(elem, score);
  }
  const std::vector<std::pair<T, int64_t>>& get() {
    resize_();
    return queue_;
  }
 private:
  void resize_() {
    std::sort(queue_.begin(), queue_.end(), [](const auto& a, const auto& b) {
      return (a.second > b.second) || (a.second == b.second && a.first < b.first);
    });
    sorted_ = true;
    if (queue_.size() > size_) queue_.resize(size_);
  }
  bool sorted_ = false;
  size_t size_ = 0;
  std::vector<std::pair<T, int64_t>> queue_;
};

static double L1DeltaByPiece(
    const sentencepiece::unigram::TrainerModel::SentencePieces& old_sp,
    const sentencepiece::unigram::TrainerModel::SentencePieces& new_sp) {
  CHECK_EQ(old_sp.size(), new_sp.size());
  const int K = static_cast<int>(old_sp.size());
  if (K == 0) return 0.0;

  // logZ(old)
  double max_old = -std::numeric_limits<double>::infinity();
  for (const auto& w : old_sp) max_old = std::max<double>(max_old, w.second);
  double sum_old = 0.0;
  for (const auto& w : old_sp) sum_old += std::exp(static_cast<double>(w.second) - max_old);
  const double logZ_old = max_old + std::log(sum_old);

  // logZ(new)
  double max_new = -std::numeric_limits<double>::infinity();
  for (const auto& w : new_sp) max_new = std::max<double>(max_new, w.second);
  double sum_new = 0.0;
  for (const auto& w : new_sp) sum_new += std::exp(static_cast<double>(w.second) - max_new);
  const double logZ_new = max_new + std::log(sum_new);

  // L1 = sum_i |p_new(i) - p_old(i)|
  double l1 = 0.0;
  for (int i = 0; i < K; ++i) {
    const double p_old = std::exp(static_cast<double>(old_sp[i].second) - logZ_old);
    const double p_new = std::exp(static_cast<double>(new_sp[i].second) - logZ_new);
    l1 += std::abs(p_new - p_old);
  }
  return l1;
}

}  // namespace

TrainerModel::TrainerModel(const TrainerSpec &trainer_spec,
                           const NormalizerSpec &normalizer_spec)
    : trainer_spec_(trainer_spec), normalizer_spec_(normalizer_spec) {}

TrainerModel::~TrainerModel() {}

const TrainerModel::SentencePieces &TrainerModel::GetSentencePieces() const {
  return sentencepieces_;
}

void TrainerModel::SetSentencePieces(SentencePieces &&sentencepieces) {
  sentencepieces_ = std::move(sentencepieces);
  CHECK(!sentencepieces_.empty());

  min_score_ = FLT_MAX;
  model_proto_data_.Clear();
  model_proto_ = &model_proto_data_;
  std::vector<std::pair<absl::string_view, int>> pieces;

  for (size_t i = 0; i < sentencepieces_.size(); ++i) {
    const absl::string_view w = sentencepieces_[i].first;  // piece
    const float score = sentencepieces_[i].second;        // score.
    CHECK(!std::isnan(score)) << "Score is NaN! Piece: " << w;
    pieces.emplace_back(w, i);
    min_score_ = std::min(min_score_, score);
    auto *piece = model_proto_data_.add_pieces();
    piece->set_piece(w.data(), w.size());
    piece->set_score(score);
  }

  BuildTrie(&pieces);
  CHECK(status().ok());
}

TrainerModel::SentencePieces Trainer::MakeSeedSentencePieces() {
  return trainer_spec_.train_extremely_large_corpus()
             ? MakeSeedSentencePiecesInternal<int64_t>()
             : MakeSeedSentencePiecesInternal<int32_t>();
}


template <typename node_int_type>
TrainerModel::SentencePieces Trainer::MakeSeedSentencePiecesInternal() {
  TrainerModel::SentencePieces seed_sentencepieces;

  // Branch A: if a seed file is provided, behave exactly like your fixed-vocab path.
  if (!trainer_spec_.seed_sentencepieces_file().empty()) {
    LOG(INFO) << "Loading pieces from seed file (dedup; drop UNK)…";
    auto input = sentencepiece::filesystem::NewReadableFile(
        trainer_spec_.seed_sentencepieces_file());
    CHECK(input != nullptr) << "Cannot open: "
                            << trainer_spec_.seed_sentencepieces_file();

    absl::flat_hash_map<std::string, char> seen;  // use as a set
    const std::string unk = trainer_spec_.unk_piece().empty()
                                ? std::string("<unk>")
                                : trainer_spec_.unk_piece();

    std::string line;
    while (input->ReadLine(&line)) {
      if (line.empty()) continue;
      const std::vector<std::string> fields = absl::StrSplit(line, '\t');
      if (fields.size() != 2) {
        LOG(WARNING) << "Skipping invalid seed line: " << line;
        continue;
      }
      const std::string& piece = fields[0];
      if (piece == unk) {
        LOG(WARNING) << "Dropping UNK from seed: " << piece;
        continue;
      }
      if (seen.emplace(piece, 1).second) {
        float score = 0.0f;
        // be tolerant of parse errors
        if (!ParseFloat(fields[1], &score)) {
          score = 0.0f;  // tolerant fallback
        }
        seed_sentencepieces.emplace_back(piece, score);
      }
    }

    LOG(INFO) << "Initialized " << seed_sentencepieces.size()
              << " seed sentencepieces (deduped; UNK excluded).";
    return seed_sentencepieces;
  }

  // Branch B: no seed file.
  // If num_sub_iterations == 0, build a seed vocabulary from the corpus
  // using the original suffix-array seeding and save it (no EM).
  // --- inside Trainer::MakeSeedSentencePiecesInternal<node_int_type>() ---
  if (trainer_spec_.num_sub_iterations() == 0 &&
      trainer_spec_.seed_sentencepieces_file().empty()) {
    TrainerModel::SentencePieces seed_sentencepieces;

    // Build UTF-32 stream + unigram counts from the corpus (as before).
    const auto* pretokenizer = SentencePieceTrainer::GetPretokenizerForTraining();
    auto pretokenize_or_rewrite = [&](std::pair<std::string, int64_t>* w) {
      if (pretokenizer) {
        std::vector<char32> chars;
        for (const auto& t : pretokenizer->PreTokenize(w->first)) {
          for (const auto& c : string_util::UTF8ToUnicodeText(t)) chars.push_back(c);
          chars.push_back(kSentenceBoundary);
        }
        return chars;
      } else if (!trainer_spec_.pretokenization_delimiter().empty()) {
        std::vector<char32> chars;
        absl::string_view delim = trainer_spec_.pretokenization_delimiter();
        for (const auto& part : absl::StrSplit(w->first, delim)) {
          for (const auto& c : string_util::UTF8ToUnicodeText(part)) chars.push_back(c);
          chars.push_back(kSentenceBoundary);
        }
        w->first = absl::StrReplaceAll(w->first, {{delim, ""}});
        return chars;
      }
      return string_util::UTF8ToUnicodeText(w->first);
    };

    std::vector<char32> array;
    absl::flat_hash_map<std::string, int64_t> all_chars;  // UTF-8 char -> freq
    const bool is_tsv = trainer_spec_.input_format() == "tsv";

    for (auto& w : sentences_) {
      const auto ut = pretokenize_or_rewrite(&w);
      for (const auto& c : ut) {
        array.push_back(c);
        if (c != kSentenceBoundary) {
          all_chars[string_util::UnicodeCharToUTF8(c)] += w.second;
        }
      }
      array.push_back(kSentenceBoundary);
      if (is_tsv) {  // oversample (parity with upstream)
        for (const auto& c : ut) array.push_back(c);
        array.push_back(kSentenceBoundary);
      }
    }

    // ---- Budgeting: vocab_size = meta + required_chars + SA_substrings ----
    const int vocab_size = trainer_spec_.vocab_size();
    const int meta_n = static_cast<int>(meta_pieces_.size());  // UNK/BOS/EOS/etc.
    int budget = vocab_size - meta_n;
    CHECK_GT(budget, 0) << "vocab_size too small after accounting for meta pieces.";

    // 1) Fill required_chars_ first (alphabet from character_coverage).
    //    We score alphabet chars by their corpus frequency for a sensible ordering.
    struct CharFreq { std::string ch; int64_t freq; };
    std::vector<CharFreq> alphabet;
    alphabet.reserve(required_chars_.size());
    for (const auto& w : Sorted(required_chars_)) {
      const std::string s = string_util::UnicodeCharToUTF8(w.first);
      const auto it = all_chars.find(s);
      const int64_t f = (it == all_chars.end()) ? 1 : it->second;
      alphabet.push_back({s, f});
    }
    // Highest freq first (stable with lexicographic tie-break in Sorted).
    std::sort(alphabet.begin(), alphabet.end(),
              [](const CharFreq& a, const CharFreq& b) {
                return (a.freq > b.freq) || (a.freq == b.freq && a.ch < b.ch);
              });

    if (static_cast<int>(alphabet.size()) > budget) {
      LOG(WARNING) << "required_chars_ (" << alphabet.size()
                   << ") exceeds available budget (" << budget
                   << "). Trimming least frequent required chars to fit.";
      alphabet.resize(budget);
    }

    absl::flat_hash_map<std::string, char> taken;  // to avoid duplicates
    for (const auto& cf : alphabet) {
      seed_sentencepieces.emplace_back(cf.ch, static_cast<float>(cf.freq));
      taken.emplace(cf.ch, 1);
    }
    int remaining = budget - static_cast<int>(seed_sentencepieces.size());
    if (remaining <= 0) {
      ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());
      LOG(INFO) << "Initialized " << seed_sentencepieces.size()
                << " seed sentencepieces (alphabet-only; budget filled).";
      return seed_sentencepieces;
    }

    // 2) Fill the remainder with frequent substrings via suffix array.
    CHECK_LE(array.size(),
             static_cast<size_t>(std::numeric_limits<node_int_type>::max()))
        << "Input corpus too large, try with train_extremely_large_corpus=true";
    const node_int_type n = static_cast<node_int_type>(array.size());
    std::vector<node_int_type> SA(n), L(n), R(n), D(n);

    constexpr node_int_type kAlphabetSize = 0x110000;  // UCS-4
    node_int_type node_num = 0;
    LOG(INFO) << "Making suffix array…";
    CHECK_EQ(0, esaxx(array.begin(), SA.begin(), L.begin(), R.begin(),
                      D.begin(), n, kAlphabetSize, node_num));

    LOG(INFO) << "Extracting frequent substrings with remaining budget: " << remaining;
    BoundedPriorityQueue<node_int_type> queue(static_cast<size_t>(remaining));

    for (node_int_type i = 0; i < node_num; ++i) {
      const node_int_type offset = SA[L[i]];
      const node_int_type len = D[i];
      if (len <= 1 || static_cast<size_t>(offset + len) >= array.size()) continue;

      const char32* begin = &array[offset];
      const char32* end   = &array[offset + len];
      if (std::find(begin, end, kSentenceBoundary) != end) continue;

      const UnicodeText uw(begin, end);
      if (!IsValidSentencePiece(uw)) continue;

      const std::string w = string_util::UnicodeTextToUTF8(uw);
      if (taken.find(w) != taken.end()) continue;  // skip alphabet duplicates

      const node_int_type freq = R[i] - L[i];
      const node_int_type score = freq * len;  // char-coverage score
      queue.push(i, score);
    }

    for (const auto& p : queue.get()) {
      if (remaining <= 0) break;
      const node_int_type offset = SA[L[p.first]];
      const node_int_type len = D[p.first];
      const char32* begin = &array[offset];
      const char32* end   = &array[offset + len];
      const UnicodeText uw(begin, end);
      const std::string w = string_util::UnicodeTextToUTF8(uw);
      if (taken.emplace(w, 1).second) {
        seed_sentencepieces.emplace_back(w, static_cast<float>(p.second));
        --remaining;
      }
    }

    ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());
    LOG(INFO) << "Initialized " << seed_sentencepieces.size()
              << " seed sentencepieces (alphabet+" << (budget - static_cast<int>(alphabet.size()))
              << " SA substrings), meta=" << meta_n
              << ", final target vocab_size=" << vocab_size;
    return seed_sentencepieces;
  }

  // Otherwise (no seed file, but EM was requested): keep your invariant.
  CHECK(false) << "No seed file provided. For corpus-based seeding set "
                  "--num_sub_iterations=0 (no EM).";
  return seed_sentencepieces;  // unreachable
}


std::vector<float> Trainer::RunEStep(const TrainerModel &model, float *obj,
                                     int64_t *num_tokens) const {
  std::vector<std::vector<float>> expected(trainer_spec_.num_threads());
  std::vector<float> objs(trainer_spec_.num_threads(), 0.0);
  std::vector<int64_t> ntokens(trainer_spec_.num_threads(), 0.0);
  auto pool = std::make_unique<ThreadPool>(trainer_spec_.num_threads());
  pool->StartWorkers();
  int64_t all_sentence_freq = 0;
  for (const auto &w : sentences_) all_sentence_freq += w.second;

  for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
    pool->Schedule([&, n]() {
      Lattice lattice;
      expected[n].resize(model.GetPieceSize(), 0.0);
      for (size_t i = n; i < sentences_.size(); i += trainer_spec_.num_threads()) {
        const std::string &w = sentences_[i].first;
        const int64_t freq = sentences_[i].second;
        lattice.SetSentence(w);
        model.PopulateNodes(&lattice);
        const float Z = lattice.PopulateMarginal(freq, &expected[n]);
        ntokens[n] += lattice.Viterbi().first.size() * freq;
        CHECK(!std::isnan(Z)) << "likelihood is NAN. Input sentence may be too long";
        objs[n] -= Z / all_sentence_freq;
      }
    });
  }
  pool.reset(nullptr);

  for (int n = 1; n < trainer_spec_.num_threads(); ++n) {
    objs[0] += objs[n];
    ntokens[0] += ntokens[n];
    for (size_t k = 0; k < expected[0].size(); ++k) {
      expected[0][k] += expected[n][k];
    }
  }

  *obj = objs[0];
  *num_tokens = ntokens[0];
  CHECK(!std::isnan(*obj));
  return expected[0];
}


TrainerModel::SentencePieces Trainer::RunMStep(
    const TrainerModel &model, const std::vector<float> &expected) const {
  const auto &sentencepieces = model.GetSentencePieces();
  CHECK_EQ(sentencepieces.size(), expected.size());
  const int K = static_cast<int>(sentencepieces.size());

  // Smoothed MLE (MAP) with symmetric Dirichlet prior α
  // NOTE - this is differenent than the Bayesian approach used by the original
  // sentencepiece library
  double alpha = 1e-1;  
  if (const char* env = std::getenv("SP_FIXED_VOCAB_ALPHA")) {
    errno = 0; char* end = nullptr; double v = std::strtod(env, &end);
    if (end != env && errno == 0 && std::isfinite(v) && v > 0.0) alpha = v;
  }

  // Sum counts (guard negatives/NaNs)
  double sum_counts = 0.0;
  std::vector<double> mass(K);
  for (int i = 0; i < K; ++i) {
    double ni = static_cast<double>(expected[i]);
    if (!std::isfinite(ni) || ni < 0.0) ni = 0.0;
    mass[i] = ni;
    sum_counts += ni;
  }

  const double denom_log = std::log(sum_counts + alpha * K);

  TrainerModel::SentencePieces new_sentencepieces;
  new_sentencepieces.reserve(K);
  for (int i = 0; i < K; ++i) {
    const double numer_log = std::log(mass[i] + alpha);
    const float logp = static_cast<float>(numer_log - denom_log);
    new_sentencepieces.emplace_back(sentencepieces[i].first, logp);
  }
  return new_sentencepieces;
}


TrainerModel::SentencePieces Trainer::FinalizeSentencePieces(
    const TrainerModel &model) const {
  // Keep the vocabulary exactly as trained; meta UNK is handled by TrainerInterface.
  return model.GetSentencePieces();
}

util::Status Trainer::Train() {
  RETURN_IF_ERROR(status());
  CHECK_EQ_OR_RETURN(TrainerSpec::UNIGRAM, trainer_spec_.model_type());
  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());

  TrainerModel model(trainer_spec_, normalizer_spec_);
  RETURN_IF_ERROR(model.status());
  RETURN_IF_ERROR(LoadSentences());

  auto seed_sentencepieces = MakeSeedSentencePieces();
  model.SetSentencePieces(std::move(seed_sentencepieces));

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  const int subiters = trainer_spec_.num_sub_iterations();
  if (subiters == 0) {
    LOG(INFO) << "num_sub_iterations=0: seed-only training (no EM). Saving model.";
    final_pieces_ = FinalizeSentencePieces(model);
    return Save();
  }

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";
  desired_vocab_size_ = trainer_spec_.vocab_size();

  // Fixed-vocab EM (your current behavior).
  for (int iter = 0; iter < subiters; ++iter) {
    float objective = 0.0f;
    int64_t num_tokens = 0;
    const auto expected = RunEStep(model, &objective, &num_tokens);
    const auto& old_pieces = model.GetSentencePieces();
    auto new_sentencepieces = RunMStep(model, expected);
    const double l1_delta = L1DeltaByPiece(old_pieces, new_sentencepieces);
    const double tv = 0.5 * l1_delta;  // total variation distance
    model.SetSentencePieces(std::move(new_sentencepieces));
    LOG(INFO) << "EM sub_iter=" << iter
              << " size=" << model.GetPieceSize()
              << " obj=" << objective
              << " num_tokens=" << num_tokens
              << " num_tokens/piece="
              << (model.GetPieceSize() > 0
                    ? (1.0 * num_tokens / model.GetPieceSize()) : 0.0)
              << " L1Δ=" << l1_delta
              << " TV=" << tv;
  }

  LOG(INFO) << "Fixed vocabulary training complete.";
  final_pieces_ = FinalizeSentencePieces(model);
  return Save();
}

}  // namespace unigram
}  // namespace sentencepiece