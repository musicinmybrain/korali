deep_learner_regularizer_header = files([
  'regularizer.hpp',
  'l2.hpp',
  'l1.hpp'
])
install_headers(deep_learner_regularizer_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

deep_learner_regularizer_source = files([
  'l2.cpp',
  'l1.cpp'
])

korali_source += deep_learner_regularizer_header
korali_source += deep_learner_regularizer_source
