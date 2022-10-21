e = find_program('./.test-run.py', required: true)

test('reinforcement.learning.upswing', e,
      timeout : 3600,
      workdir: meson.current_source_dir(),
      depends: python_extension,
      suite: 'regression'
    )
