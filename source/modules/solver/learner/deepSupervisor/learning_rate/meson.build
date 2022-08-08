deep_learner_learning_rate_header = files([
  'learning_rate.hpp',
  'step_based_decay.hpp',
  'time_based_decay.hpp'
])
install_headers(deep_learner_learning_rate_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

deep_learner_learning_rate_source = files([
  'learning_rate.cpp',
  'step_based_decay.cpp',
  'time_based_decay.cpp'
])

korali_source += deep_learner_learning_rate_header
korali_source += deep_learner_learning_rate_source
