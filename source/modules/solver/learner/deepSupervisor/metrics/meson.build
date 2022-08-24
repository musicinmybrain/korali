deep_learner_metrics_header = files([
  'accuracy.hpp',
  'metrics.hpp',
])
install_headers(deep_learner_metrics_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

deep_learner_metrics_source = files([
  'accuracy.cpp',
  'metrics.cpp',
])

korali_source += deep_learner_metrics_header
korali_source += deep_learner_metrics_source
