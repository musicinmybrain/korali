e = find_program('run-ffn.py', required: true)
test('learning.supervised.sineWave.fnn.adabelief', e,
      args: [ '--optimizer', 'AdaBelief', '--learningRate', '0.005', '--maxGenerations', '30' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('learning.supervised.sineWave.fnn.adagrad', e,
      args: [ '--optimizer', 'Adagrad', '--learningRate', '0.05', '--maxGenerations', '30'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('learning.supervised.sineWave.fnn.adam', e,
      args: [ '--optimizer', 'Adam', '--learningRate', '0.005', '--maxGenerations', '30' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('learning.supervised.sineWave.fnn.madgrad', e,
      args: [ '--optimizer', 'MADGRAD', '--learningRate', '0.005', '--maxGenerations', '30'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
    
e = find_program('run-ffn.py', required: true)
test('learning.supervised.sineWave.fnn.rmsprop', e,
      args: [ '--optimizer', 'RMSProp', '--learningRate', '0.002', '--maxGenerations', '30' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('learning.supervised.sineWave.fnn.adam', e,
      args: [ '--optimizer', 'Adam', '--learningRate', '0.005', '--engine', 'Korali', '--maxGenerations', '30'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
