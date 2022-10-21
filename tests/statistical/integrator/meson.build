e = find_program('./run-integrator-1.py', required: true)
test('samplers.correctness.integrator-1', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./run-integrator-2.py', required: true)
test('samplers.correctness.integrator-2', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./run-integrator-3.py', required: true)
test('samplers.correctness.integrator-3', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
    
e = find_program('./run-integrator-4.py', required: true)
test('samplers.correctness.integrator-4', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
