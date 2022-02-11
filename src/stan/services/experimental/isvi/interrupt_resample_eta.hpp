#ifndef STAN_ISVI_INTERRUPT_RESAMPLE_ETA_HPP
#define STAN_ISVI_INTERRUPT_RESAMPLE_ETA_HPP

#include <stan/callbacks/interrupt.hpp>
#include "isvi_stams_wrapper.hpp"

namespace stan {
namespace isvi {

/**
 * This class holds on to a reference to a isvi_stams_model_wrapper.
 * It is an <code>interrupt</code> that is called by the sampler
 * at the start of each sample's loop. We use that as an opportunity
 * to modify the 'eta' values inside the isvi_stams_model_wrapper.
 * 
 * Note: if isvi_stams_model_wrapper was created with stochastic=false,
 * this does nothing.
 */
template <class Model, class BaseRNG>
class interrupt_resample_eta : public stan::callbacks::interrupt {
public:
  interrupt_resample_eta(isvi_stams_model_wrapper<Model, BaseRNG>& isvi_model)
  : stan::callbacks::interrupt(), isvi_model_(isvi_model) {}

  /**
   * This implements the callback function, once per sample.
   */
  void operator()() override {
    if (isvi_model_.stochastic()) {
      isvi_model_.resample_eta();
    }
  }

protected:
  isvi_stams_model_wrapper<Model, BaseRNG>& isvi_model_;
};

}  // namespace isvi
}  // namespace stan

#endif