#ifndef STAN_CALLBACKS_CHAIN_INTERRUPT_HPP
#define STAN_CALLBACKS_CHAIN_INTERRUPT_HPP

#include <stan/callbacks/interrupt.hpp>
#include <iostream>

namespace stan {
namespace callbacks {

/**
 * <code>chain_interrupt</code> inherits from <code>interrupt</code>
 * but also wraps two existing <code>interrupt</code>s.
 * 
 * When the 'chain' interrupt is called, it calls first() followed
 * by second().
 */
class chain_interrupt : public interrupt {
 public:
  chain_interrupt(interrupt& first, interrupt& second)
  : interrupt(), first_(first), second_(second) {}

  /**
   * Callback function.
   *
   * This function is called by the algorithms allowing the interfaces
   * to break when necessary.
   */
  void operator()() {
    first_();
    second_();
  }

protected:
  interrupt& first_;
  interrupt& second_;
};

}  // namespace callbacks
}  // namespace stan
#endif
