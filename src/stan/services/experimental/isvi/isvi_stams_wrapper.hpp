#ifndef STAN_ISVI_STAMS_MODEL_WRAPPER_HPP
#define STAN_ISVI_STAMS_MODEL_WRAPPER_HPP

#include <stan/math.hpp>
#include <stan/model/model_base_crtp.hpp>
#include <stan/callbacks/logger.hpp>
#include <stdexcept>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace isvi {

/** ------------------------------------------------------------------
 * TEMPORARY - GETTING AROUND TYPING ISSUES WITH NORMAL_MEANFIELD BY 
 * REIMPLEMENTING THE KEY FUNCTIONS WE NEED FROM IT HERE WITH TEMPLATE<T>
 */
template <typename T>
inline T q_entropy(Eigen::Matrix<T, -1, 1>& params_r, int dim){
    auto log_sigma_ = params_r.tail(dim);
    double d = static_cast<double>(dim);
    return 0.5 * d * (1.0 + stan::math::LOG_TWO_PI) + log_sigma_.sum();
  }

template <typename T>
inline T q_log_det_fisher(Eigen::Matrix<T, -1, 1>& params_r, int dim){
  auto log_sigma_ = params_r.tail(dim);
  return -2.0 * log_sigma_.sum();
}

template <typename T>
inline Eigen::Matrix<T, -1, 1> q_sample(const Eigen::Matrix<T, -1, 1>& params_r, int dim, const Eigen::VectorXd& eta){
  // Given 'eta', which is a single sample from dim-dimensional normal(0,1),
  // transform it into a sample from q.
  if(eta.size() != dim)
    throw new std::runtime_error("stan::isvi::q_sample wrong size for eta!");
  
  auto mu_ = params_r.head(dim);
  auto sigma_ = stan::math::exp(params_r.tail(dim));
  return mu_ + sigma_.cwiseProduct(eta);
}
/** ------------------------------------------------------------------ */

/** Model wrapper class derives from the base class for models, i.e. stan::model::model_base_crtp.
 * It wraps a given model with a variational approximation, and provides "log prob" gradients
 * for the variational parameters.
 * 
 * Note on CRTP pattern: the full typename of this class is isvi_stams_model_wrapper<Model,BaseRNG>,
 * and in the CRTP pattern we make class Foo inherit from Bar<Foo>. Here, 'Bar' is model_base_crtp,
 * and 'Foo' is isvi_stams_model_wrapper<Model,BaseRNG>. Hence some verbose nested templates.
 */
template <class Model, class BaseRNG>
class isvi_stams_model_wrapper : public stan::model::model_base_crtp<isvi_stams_model_wrapper<Model, BaseRNG>> {
 public:
  isvi_stams_model_wrapper(Model& m, BaseRNG& rng, int n_monte_carlo_kl, double lambda, bool stochastic, double min_omega, double max_omega)
  : stan::model::model_base_crtp<isvi_stams_model_wrapper<Model, BaseRNG>>(2*m.num_params_r()),
  wrapped_(m),
  rng_(rng),
  n_monte_carlo_kl_(n_monte_carlo_kl),
  lambda_(lambda),
  stochastic_(stochastic),
  min_omega_(min_omega),
  max_omega_(max_omega),
  presampled_eta_(n_monte_carlo_kl, m.num_params_r()){
    // Sanity checks on inputs
    static const char* function = "stan::isvi::isvi_stams_model_wrapper";
    math::check_positive(function,
      "Number of Monte Carlo samples for KL",
      n_monte_carlo_kl_);
    math::check_nonnegative(function,
      "Lambda must be greater than or equal to 1",
      lambda - 1.0);
    math::check_nonnegative(function,
      "max_omega must be larger than min_omega",
      max_omega - min_omega);
    math::check_finite(function,
      "-Infinite min_omega not yet supported",
      min_omega);
    math::check_finite(function,
      "+Infinite max_omega not yet supported",
      max_omega);

    resample_eta();
  }

  void resample_eta() {
    for(int i=0; i<n_monte_carlo_kl_; ++i){
      for(int j=0; j<wrapped_.num_params_r(); ++j){
        presampled_eta_(i, j) = stan::math::normal_rng(0, 1, rng_);
      }
    }
  }

  bool stochastic() {
    return stochastic_;
  }

  // ===== BEGIN BLOCK COPIED FROM ADVI =====

  /**
   * Modified copy of calc_ELBO from ADVI
   */
  template<typename T>
  T kl_q_p(Eigen::Matrix<T,-1,1>& theta) const {
    static const char* function = "stan::isvi::kl_q_p";
    
    T cross_entropy = 0.0;
    int dim = wrapped_.num_params_r();
    int n_dropped_evaluations = 0;
    int n_succeeded_evaluations = 0;
    for (int i = 0; i < n_monte_carlo_kl_; ++i) {
      auto zeta = q_sample<T>(theta, dim, presampled_eta_.row(i));
      try {
        std::stringstream ss;
        T log_prob = wrapped_.template log_prob<false, true, T>(zeta, &ss);
        stan::math::check_finite(function, "log_prob", log_prob);
        cross_entropy -= log_prob;
        n_succeeded_evaluations++;
      } catch (const std::domain_error& e) {
        n_dropped_evaluations++;
      }
    }

    if (n_succeeded_evaluations == 0) {
      const char* name = "The number of dropped evaluations";
      const char* msg1 = "has reached its maximum amount (";
      const char* msg2 = ").";
      stan::math::throw_domain_error(function, name, n_dropped_evaluations, msg1, msg2);
    }

    // Denominator of average is number of *successful* points
    cross_entropy = cross_entropy / n_succeeded_evaluations;
    // KL = CE - H
    return cross_entropy - q_entropy<T>(theta, dim);
  }
  // ===== END BLOCK COPIED FROM ADVI =====


  // ===== BEGIN OVERRIDING MODEL_BASE_CRTP DEFINITIONS =====
  /**
   * Return the name of the model.
   *
   * @return model name
   */
  std::string model_name() const override {
    return "isvi_lambda" + std::to_string(lambda_) + "_" + wrapped_.model_name();
  }

  /**
   * Returns the compile information of the model:
   * stanc version and stanc flags used to compile the model.
   *
   * @return model name
   */
  std::vector<std::string> model_compile_info() const override {
    return wrapped_.model_compile_info();
  }

  /**
   * Set the specified argument to sequence of parameters, transformed
   * parameters, and generated quantities in the order in which they
   * were declared.  The input sequence is cleared and resized.
   *
   * @param[in,out] names sequence of names parameters, transformed
   * parameters, and generated quantities
   */
  void get_param_names(std::vector<std::string>& names) const override {
    std::vector<std::string> wrapped_names;
    wrapped_.get_param_names(wrapped_names);
    names.resize(0);
    for (std::string dim_name : wrapped_names)
      names.push_back("mu_" + dim_name);
    for (std::string dim_name : wrapped_names)
      names.push_back("omega_" + dim_name);
  }

  /**
   * Set the dimensionalities of constrained parameters, transformed
   * parameters, and generated quantities.  The input sequence is
   * cleared and resized.  The dimensions of each parameter
   * dimensionality is represented by a sequence of sizes.  Scalar
   * real and integer parameters are represented as an empty sequence
   * of dimensions.
   *
   * <p>Indexes are output in the order they are used in indexing. For
   * example, a 2 x 3 x 4 array will have dimensionality
   * `std::vector<size_t>{ 2, 3, 4 }`, whereas a 2-dimensional array
   * of 3-vectors will have dimensionality `std::vector<size_t>{ 2, 3
   * }`, and a 2-dimensional array of 3 x 4 matrices will have
   * dimensionality `std::vector<size_t>{2, 3, 4}`.
   *
   * @param[in,out] dimss sequence of dimensions specifications to set
   */
  void get_dims(std::vector<std::vector<size_t> >& dimss) const override {
    dimss.resize(0);
    // We have only unconstrained parameters, so num constrained = num unconstrained
    dimss.emplace_back(std::vector<size_t>{this->num_params_r()});
    // We have zero transformed parameters
    dimss.emplace_back(std::vector<size_t>{});
    // We have zero generated quantities
    dimss.emplace_back(std::vector<size_t>{});
  }

  void constrained_param_names(std::vector<std::string>& param_names,
                               bool include_tparams,
                               bool include_gqs) const override {
    // Since the variational approximation is over the *unconstrained*
    // params in the wrapped model, we still use those same names for 
    // the *constrained* params here.
    std::vector<std::string> wrapped_names;
    wrapped_.unconstrained_param_names(wrapped_names);
    // param_names.resize(0);
    for (std::string dim_name : wrapped_names)
      param_names.push_back("mu_" + dim_name);
    for (std::string dim_name : wrapped_names)
      param_names.push_back("omega_" + dim_name);
  }

  void unconstrained_param_names(std::vector<std::string>& param_names,
                                 bool include_tparams,
                                 bool include_gqs) const override {
    std::vector<std::string> wrapped_names;
    wrapped_.unconstrained_param_names(wrapped_names);
    // param_names.resize(0);
    for (std::string dim_name : wrapped_names)
      param_names.push_back("mu_" + dim_name);
    for (std::string dim_name : wrapped_names)
      param_names.push_back("omega_" + dim_name);
  }

  template <bool propto, bool jacobian, typename T>
  T log_prob(Eigen::Matrix<T, -1, 1>& params_r,
             std::ostream* msgs) const {
    T log_det_fisher = q_log_det_fisher<T>(params_r, wrapped_.num_params_r());
    T kl_qp = kl_q_p(params_r);
    // If log standard deviation (omega) falls below min_omega_ or above max_omega_,
    // this adds a gradient that nudges it back to more sensible values.
    T clip_omega_penalty = 0.0;
    // TODO - find the right Eigen operations to vectorize this.
    auto omega_ = params_r.tail(wrapped_.num_params_r());
    for (int i=0; i<omega_.size(); ++i) {
      if (omega_(i) < min_omega_) {
        clip_omega_penalty += omega_(i) - min_omega_;
      } else if (omega_(i) > max_omega_) {
        clip_omega_penalty += max_omega_ - omega_(i);
      }
    }
    return 0.5*log_det_fisher - lambda_ * kl_qp + clip_omega_penalty;
  }

  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
             std::ostream* msgs) const {
    // Cast vector -> matrix and just redirect to the above implementation
    Eigen::Matrix<T, -1, 1> eigen_params_r(params_r.size());
    for (int i=0; i<params_r.size(); ++i)
      eigen_params_r[i] = params_r[i];
    return this->template log_prob<propto, jacobian, T>(eigen_params_r, msgs);
  }

  /**
   * Read constrained parameter values from the specified context,
   * unconstrain them, then concatenate the unconstrained sequences
   * into the specified parameter sequence.  Output messages go to the
   * specified stream.
   *
   * @param[in] context definitions of variable values
   * @param[in,out] params_r unconstrained parameter values produced
   * @param[in,out] msgs stream to which messages are written
   */
  void transform_inits(const io::var_context& context, Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {
    throw new std::runtime_error("stan::isvi::isvi_stams_model_wrapper::transform_inits not implemented!");
  }

  void transform_inits(const stan::io::var_context& context, std::vector<int>& params_i,
                       std::vector<double>& params_r, std::ostream* msgs) const override {
    throw new std::runtime_error("stan::isvi::isvi_stams_model_wrapper::transform_inits not implemented!");
  }

  template <typename RNG>
  void write_array(RNG& base_rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_constrained_r, bool include_tparams,
                   bool include_gqs, std::ostream* msgs) const {
    params_constrained_r = params_r;
  }

  template <typename RNG>
  void write_array(RNG& base_rng, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::vector<double>& params_r_constrained,
                   bool include_tparams, bool include_gqs,
                   std::ostream* msgs) const {
    params_r_constrained = params_r;
  }

  // ===== END OVERRIDING MODEL_BASE_CRTP =====

 protected:
  Model& wrapped_;
  BaseRNG& rng_;
  int n_monte_carlo_kl_;
  double lambda_;
  bool stochastic_;
  double min_omega_;
  double max_omega_;
  Eigen::MatrixXd presampled_eta_;
};

}  // namespace isvi
}  // namespace stan

#endif