deep_learner_reward_header = files([
  'negative_log_likelihood.hpp',
  'cross_entropy.hpp',
  'mse.hpp',
  'reward.hpp',
])
install_headers(deep_learner_reward_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

deep_learner_reward_source = files([
  'negative_log_likelihood.cpp',
  'cross_entropy.cpp',
  'mse.cpp',
  'reward.cpp',
])

korali_source += deep_learner_reward_header
korali_source += deep_learner_reward_source
