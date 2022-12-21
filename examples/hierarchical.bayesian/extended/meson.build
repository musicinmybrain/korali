e = find_program('./.test-run.py', required: true)
test('hierarchical.bayesian.extended', e,
      timeout : 5000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc,
      priority : 1000
    )
