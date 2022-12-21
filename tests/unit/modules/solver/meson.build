optimizers_tester = executable('optimizers_tester',
  files(['optimizers.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.solver.optimizers', optimizers_tester,
  protocol: 'gtest',
  suite: 'unit',
)

samplers_tester = executable('samplers_tester',
  files(['samplers.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.solver.samplers', samplers_tester,
  protocol: 'gtest',
  suite: 'unit'
)

executor_tester = executable('executor_tester',
  files(['executor.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.solver.executor', executor_tester,
  protocol: 'gtest',
  suite: 'unit',
)

integrator_tester = executable('integrator_tester',
  files(['integrator.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.solver.integrator', integrator_tester,
  protocol: 'gtest',
  suite: 'unit',
)

deepSupervisor_tester = executable('deepSupervisor_tester',
  files(['deepSupervisor.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.solver.deepSupervisor', deepSupervisor_tester,
  protocol: 'gtest',
  suite: 'unit',
)

agent_tester = executable('agent_tester',
  files(['agent.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.solver.agent', agent_tester,
  protocol: 'gtest',
  suite: 'unit',
)

