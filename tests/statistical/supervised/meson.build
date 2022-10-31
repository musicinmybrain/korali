e = find_program('./run-deep-supervisor.py', required: true)
test('learning.supervised.deep-supervisor', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

