#ifndef STAN_ISVI_STAMS_MODEL_WRAPPER_HPP
#define STAN_ISVI_STAMS_MODEL_WRAPPER_HPP

#include <stan/math.hpp>
#include <stan/model/model_base_crtp.hpp>
#include <stan/callbacks/logger.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace isvi {

/** Model wrapper class derives from the base class for models, i.e. stan::model::model_base_crtp.
 * It wraps a given model with a variational approximation, and provides "log prob" gradients
 * for the variational parameters.
 * 
 * Note on CRTP pattern: the full typename of this class is isvi_stams_model_wrapper<Model,Q,BaseRNG>,
 * and in the CRTP pattern we make class Foo inherit from Bar<Foo>. Here, 'Bar' is model_base_crtp,
 * and 'Foo' is isvi_stams_model_wrapper<Model,Q,BaseRNG>. Hence some verbose nested templates.
 */
template <class Model, class Q, class BaseRNG>
class isvi_stams_model_wrapper : public stan::model::model_base_crtp<isvi_stams_model_wrapper<Model, Q, BaseRNG>> {
 public:
  isvi_stams_model_wrapper(Model& m, BaseRNG& rng, int n_monte_carlo_grad, int n_monte_carlo_kl, double lambda)
  : stan::model::model_base_crtp<isvi_stams_model_wrapper<Model, Q, BaseRNG>>(calc_n_params(m.num_params_r())),
  wrapped_(m),
  rng_(rng),
  n_monte_carlo_grad_(n_monte_carlo_grad),
  n_monte_carlo_kl_(n_monte_carlo_kl),
  lambda_(lambda) {
    // Sanity checks on inputs
    static const char* function = "stan::isvi::isvi_stams";
    math::check_positive(function,
      "Number of Monte Carlo samples for gradients",
      n_monte_carlo_grad_);
    math::check_positive(function,
      "Number of Monte Carlo samples for KL",
      n_monte_carlo_kl_);
    math::check_positive(function,
      "Lambda must be greater than or equal to 1.0",
      lambda - 1.0);
  }

  // ===== BEGIN BLOCK COPIED FROM ADVI =====

  /**
   * Calculates the Evidence Lower BOund (ELBO) by sampling from
   * the variational distribution and then evaluating the log joint,
   * adjusted by the entropy term of the variational distribution.
   *
   * @param[in] q variational approximation at which to evaluate
   * the ELBO.
   * @return the evidence lower bound.
   * @throw std::domain_error If, after n_monte_carlo_kl_ number of draws
   * from the variational distribution all give non-finite log joint
   * evaluations. This means that the model is severely ill conditioned or
   * that the variational distribution has somehow collapsed.
   */
  double calc_ELBO(const Q& q) const {
    static const char* function = "stan::isvi::calc_ELBO";

    double elbo = 0.0;
    int dim = q.dimension();
    Eigen::VectorXd zeta(dim);

    int n_dropped_evaluations = 0;
    for (int i = 0; i < n_monte_carlo_kl_;) {
      q.sample(rng_, zeta);
      try {
        std::stringstream ss;
        double log_prob = wrapped_.template log_prob<false, true>(zeta, &ss);
        // if (ss.str().length() > 0)
        //   logger.info(ss);
        stan::math::check_finite(function, "log_prob", log_prob);
        elbo += log_prob;
        ++i;
      } catch (const std::domain_error& e) {
        ++n_dropped_evaluations;
        if (n_dropped_evaluations >= n_monte_carlo_kl_) {
          const char* name = "The number of dropped evaluations";
          const char* msg1 = "has reached its maximum amount (";
          const char* msg2
              = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
          stan::math::throw_domain_error(function, name, n_monte_carlo_kl_,
                                         msg1, msg2);
        }
      }
    }
    elbo /= n_monte_carlo_kl_;
    elbo += q.entropy();
    return elbo;
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
    // TODO - IS THIS RIGHT? WHAT IS DIFFERENCE BETWEEN PARAMS, UNCONSTRAINED PARAMS, ETC?
    std::vector<std::string> tmp_names;
    wrapped_.get_param_names(tmp_names);
    names.resize(0);
    // TODO - create member pointer to a q to avoid creating and destroying on the fly?
    Q q(wrapped_.num_params_r());
    q.get_param_names(tmp_names, names);
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
    dimss.push_back(std::vector<size_t>{this->num_params_r()});
    // We have zero transformed parameters
    dimss.push_back(std::vector<size_t>{0});
    // We have zero generated quantities
    dimss.push_back(std::vector<size_t>{0});
  }

  void constrained_param_names(std::vector<std::string>& param_names,
                               bool include_tparams,
                               bool include_gqs) const override {}

  void unconstrained_param_names(std::vector<std::string>& param_names,
                                 bool include_tparams,
                                 bool include_gqs) const override {}

  template <bool propto, bool jacobian, typename T>
  T log_prob(Eigen::Matrix<T, -1, 1>& params_r, std::ostream* msgs) const {
    // TODO - create member pointer to a q to avoid creating and destroying on the fly?
    Q q(wrapped_.num_params_r());
    q.set_theta(params_r);
    T log_det_fisher = q.template log_det_fisher<T>();
    T kl_qp = -calc_ELBO(q);
    return 0.5*log_det_fisher - lambda_ * kl_qp;
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
  void transform_inits(const io::var_context& context,
                       Eigen::VectorXd& params_r,
                       std::ostream* msgs) const override {
    // TODO - anything we need to do with constrained/unconstrained here?
  }

  template <typename RNG>
  void write_array(RNG& base_rng, Eigen::VectorXd& params_r,
                   Eigen::VectorXd& params_constrained_r, bool include_tparams,
                   bool include_gqs, std::ostream* msgs) const {}

  template <bool propto, bool jacobian, typename T>
  T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
             std::ostream* msgs) const {
    (*msgs) << "WTF: log_prob vector" << std::endl;
    return 0;
  }

  void transform_inits(const stan::io::var_context& context,
                       std::vector<int>& params_i,
                       std::vector<double>& params_r,
                       std::ostream* msgs) const override {
    (*msgs) << "WTF: transform_inits vector" << std::endl;
  }

  template <typename RNG>
  void write_array(RNG& base_rng, std::vector<double>& params_r,
                   std::vector<int>& params_i,
                   std::vector<double>& params_r_constrained,
                   bool include_tparams, bool include_gqs,
                   std::ostream* msgs) const {
    (*msgs) << "WTF: write_array vector" << std::endl;
  }

  // ===== END OVERRIDING MODEL_BASE_CRTP =====

 protected:
  Model& wrapped_;
  BaseRNG& rng_;
  int n_monte_carlo_grad_;
  int n_monte_carlo_kl_;
  double lambda_;
 
 static int calc_n_params(int dims) {
    Q q(dims);
    return q.parameters();
  }
};

}  // namespace isvi
}  // namespace stan

#endif