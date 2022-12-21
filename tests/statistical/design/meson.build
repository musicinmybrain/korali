e = find_program('./run-design-1.py', required: true)
test('solvers.correctness.design-1', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./run-design-2.py', required: true)
test('solvers.correctness.design-2', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./run-design-3.py', required: true)
test('solvers.correctness.design-3', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
