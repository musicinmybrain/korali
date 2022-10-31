e = find_program('./.test-run.py', required: true)
test('reinforcement.learning.cartpole', e,
      timeout : 5000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
