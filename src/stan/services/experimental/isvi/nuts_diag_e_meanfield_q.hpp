#ifndef STAN_SERVICES_EXPERIMENTAL_ISVI_MEANFIELD_HPP
#define STAN_SERVICES_EXPERIMENTAL_ISVI_MEANFIELD_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/experimental_message.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/io/var_context.hpp>
#include <stan/services/experimental/isvi/isvi_stams_wrapper.hpp>
#include <boost/random/additive_combine.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/services/util/run_sampler.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/inv_metric.hpp>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace experimental {
namespace isvi {

/**
 * Runs ISVI algorithm with mean-field (diagonal Gaussian) Q
 * 
 * Based on adapting advi/meanfield service and sample/hmc_nuts_diag_e
 * services
 *
 * @tparam Model A model implementation
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] init var context for initialization
 * @param[in] init_inv_metric var context exposing an initial diagonal
 *            inverse Euclidean metric (must be positive definite)
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the random number generator
 * @param[in] init_radius radius to initialize
 * ---------- NUTS-like arguments ----------
 * @param[in] num_warmup Number of warmup samples
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] save_warmup Indicates whether to save the warmup iterations
 * @param[in] refresh Controls the output
 * @param[in] stepsize initial stepsize for discrete evolution
 * @param[in] stepsize_jitter uniform random jitter of stepsize
 * @param[in] max_depth Maximum tree depth
 * ---------- ADVI-like arguments ----------
 * @param[in] grad_samples number of samples for Monte Carlo estimate of gradients
 * @param[in] kl_samples number of samples for Monte Carlo estimate of kl
 * @param[in] lambda controls Sampling/VI trade-off
 * ---------- Further common arguments ----------
 * @param[in,out] interrupt callback to be called every sample
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] sample_writer output for samples of parameter values
 * @param[in,out] diagnostic_writer output for diagnostic values
 * @return error_codes::OK if successful
 */
template <class Model>
int nuts_diag_e_meanfield_q(Model& model, const stan::io::var_context& init,
                            const stan::io::var_context& init_inv_metric,
                            unsigned int random_seed, unsigned int chain,
                            double init_radius, int num_warmup, int num_samples,
                            int num_thin, bool save_warmup, int refresh,
                            double stepsize, double stepsize_jitter, int max_depth,
                            int grad_samples, int kl_samples, double lambda,
                            callbacks::interrupt& interrupt,
                            callbacks::logger& logger, 
                            callbacks::writer& init_writer,
                            callbacks::writer& sample_writer,
                            callbacks::writer& diagnostic_writer) {
  util::experimental_message(logger);

  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  // Whereas 'model' is the compiled STAN model, 'wrapped_model' is the model we
  // will pass to the sampler.
  stan::isvi::isvi_stams_model_wrapper<Model, stan::variational::normal_meanfield, boost::ecuyer1988>
    wrapped_model(model, rng, grad_samples, kl_samples, lambda);

  std::vector<double> cont_vector = util::initialize(
      wrapped_model, init, rng, init_radius, true, logger, init_writer);

  Eigen::VectorXd inv_metric;
  try {
    inv_metric = util::read_diag_inv_metric(init_inv_metric,
                                            wrapped_model.num_params_r(),
                                            logger);
    util::validate_diag_inv_metric(inv_metric, logger);
  } catch (const std::domain_error& e) {
    return error_codes::CONFIG;
  }

  stan::mcmc::diag_e_nuts<Model, boost::ecuyer1988> sampler(wrapped_model, rng);

  sampler.set_metric(inv_metric);
  sampler.set_nominal_stepsize(stepsize);
  sampler.set_stepsize_jitter(stepsize_jitter);
  sampler.set_max_depth(max_depth);

  util::run_sampler(sampler, wrapped_model, cont_vector, num_warmup, num_samples,
                    num_thin, refresh, save_warmup, rng, interrupt, logger,
                    sample_writer, diagnostic_writer);

  return error_codes::OK;
}


// Overload - this one called by command
template <class Model>
int nuts_diag_e_meanfield_q(Model& model, const stan::io::var_context& init,
                            unsigned int random_seed, unsigned int chain,
                            double init_radius, int num_warmup, int num_samples,
                            int num_thin, bool save_warmup, int refresh,
                            double stepsize, double stepsize_jitter, int max_depth,
                            int grad_samples, int kl_samples, double lambda,
                            callbacks::interrupt& interrupt,
                            callbacks::logger& logger, 
                            callbacks::writer& init_writer,
                            callbacks::writer& sample_writer,
                            callbacks::writer& diagnostic_writer) {
  stan::io::dump dmp
      = util::create_unit_e_diag_inv_metric(model.num_params_r());
  stan::io::var_context& unit_e_metric = dmp;

  return nuts_diag_e_meanfield_q(model, init, unit_e_metric, random_seed, chain, init_radius,
    num_warmup, num_samples, num_thin, save_warmup, refresh, stepsize, stepsize_jitter, max_depth,
    grad_samples, kl_samples, lambda,
    interrupt, logger, init_writer, sample_writer, diagnostic_writer);
}

}  // namespace isvi
}  // namespace experimental
}  // namespace services
}  // namespace stan
#endif
