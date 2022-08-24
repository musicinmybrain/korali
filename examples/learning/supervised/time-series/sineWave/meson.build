e = find_program('run-rnn.py', required: true)
test('learning.supervised.sineWave.rnn.adabelief', e,
      args: [ '--optimizer', 'AdaBelief', '--learningRate', '0.001', '--maxGenerations', '60' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-rnn.py', required: true)
test('learning.supervised.sineWave.rnn.adagrad', e,
      args: [ '--optimizer', 'Adagrad', '--learningRate', '0.01', '--maxGenerations', '60' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-rnn.py', required: true)
test('learning.supervised.sineWave.rnn.adam', e,
      args: [ '--optimizer', 'Adam', '--learningRate', '0.001', '--maxGenerations', '60' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-rnn.py', required: true)
test('learning.supervised.sineWave.rnn.madgrad', e,
      args: [ '--optimizer', 'MADGRAD', '--learningRate', '0.001', '--maxGenerations', '60'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-rnn.py', required: true)
test('learning.supervised.sineWave.rnn.rmsprop', e,
      args: [ '--optimizer', 'RMSProp', '--learningRate', '0.001', '--maxGenerations', '60'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-rnn.py', required: true)
test('learning.supervised.sineWave.rnn.lstm', e,
      args: [ '--optimizer', 'Adam', '--learningRate', '0.001', '--rnnType', 'LSTM', '--maxGenerations', '60' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
