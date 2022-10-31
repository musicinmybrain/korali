e = find_program('./.test-run.py', required: true)
test('supervised.learning.mnist', e,
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
