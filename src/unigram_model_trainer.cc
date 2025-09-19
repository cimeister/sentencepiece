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
  return MakeSeedSentencePiecesInternal<int32_t>();
}

// Simple seed loader: dedup, drop UNK (SP will add meta UNK).
template <typename node_int_type>
TrainerModel::SentencePieces Trainer::MakeSeedSentencePiecesInternal() {
  TrainerModel::SentencePieces seed_sentencepieces;
  CHECK(!trainer_spec_.seed_sentencepieces_file().empty())
      << "Requires --seed_sentencepieces_file to be set.";

  LOG(INFO) << "Loading pieces from seed file (dedup; drop UNK)...";

  auto input = sentencepiece::filesystem::NewReadableFile(
      trainer_spec_.seed_sentencepieces_file());
  CHECK(input != nullptr) << "Cannot open: "
                          << trainer_spec_.seed_sentencepieces_file();

  absl::flat_hash_map<std::string, char> seen;  // use as a set
  const std::string unk =
      trainer_spec_.unk_piece().empty() ? std::string("<unk>")
                                        : trainer_spec_.unk_piece();

  std::string line;
  while (input->ReadLine(&line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = absl::StrSplit(line, '\t');
    if (fields.size() != 2) {
      LOG(WARNING) << "Skipping invalid seed line: " << line;
      continue;
    }
    const std::string &piece = fields[0];
    if (piece == unk) {
      // Drop UNK from seed; SP will add meta UNK.
      LOG(WARNING) << "Dropping UNK from seed: " << piece;
      continue;
    }
    if (seen.emplace(piece, 1).second) {
      float score = 0.0f;
      // Be tolerant of parse errors.
      try { score = std::stof(fields[1]); } catch (...) { score = 0.0f; }
      seed_sentencepieces.emplace_back(piece, score);
    }
  }

  LOG(INFO) << "Initialized " << seed_sentencepieces.size()
            << " seed sentencepieces (deduped; UNK excluded).";
  return seed_sentencepieces;
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

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";
  desired_vocab_size_ = trainer_spec_.vocab_size();

  while (true) {
    for (int iter = 0; iter < trainer_spec_.num_sub_iterations(); ++iter) {
      float objective = 0.0;
      int64_t num_tokens = 0;
      const auto expected = RunEStep(model, &objective, &num_tokens);
      auto new_sentencepieces = RunMStep(model, expected);
      model.SetSentencePieces(std::move(new_sentencepieces));
      LOG(INFO) << "EM sub_iter=" << iter << " size=" << model.GetPieceSize()
                << " obj=" << objective << " num_tokens=" << num_tokens
                << " num_tokens/piece="
                << (model.GetPieceSize() > 0 ? (1.0 * num_tokens / model.GetPieceSize()) : 0.0);
    }
    LOG(INFO) << "Fixed vocabulary training complete. Breaking main EM loop.";
    break;
    // The original PruneSentencePieces call is disabled.
    // auto new_sentencepieces = PruneSentencePieces(model, desired_vocab_size_);
    // model.SetSentencePieces(std::move(new_sentencepieces));
  }

  final_pieces_ = FinalizeSentencePieces(model);
  return Save();
}
}  // namespace unigram
}  // namespace sentencepiece