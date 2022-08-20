deep_learner_loss_header = files([
  'cross_entropy.hpp',
  'mse.hpp',
  'loss.hpp',
])
install_headers(deep_learner_loss_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

deep_learner_loss_source = files([
  'cross_entropy.cpp',
  'mse.cpp',
  'loss.cpp',
])

korali_source += deep_learner_loss_header
korali_source += deep_learner_loss_source
