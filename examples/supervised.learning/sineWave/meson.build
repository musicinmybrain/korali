e = find_program('run-ffn.py', required: true)
test('supervised.learning.sineWave.fnn.adabelief', e,
      args: [ '--optimizer', 'fAdaBelief', '--learningRate', '0.005', '--maxGenerations', '30' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('supervised.learning.sineWave.fnn.adagrad', e,
      args: [ '--optimizer', 'fAdaGrad', '--learningRate', '0.05', '--maxGenerations', '30'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('supervised.learning.sineWave.fnn.adam', e,
      args: [ '--optimizer', 'fAdam', '--learningRate', '0.005', '--maxGenerations', '30' ],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('supervised.learning.sineWave.fnn.madgrad', e,
      args: [ '--optimizer', 'fMadGrad', '--learningRate', '0.005', '--maxGenerations', '30'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-ffn.py', required: true)
test('supervised.learning.sineWave.fnn.adam', e,
      args: [ '--optimizer', 'fAdam', '--learningRate', '0.005', '--engine', 'Korali', '--maxGenerations', '30'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

# e = find_program('run-rnn.py', required: true)
# test('supervised.learning.sineWave.rnn.adabelief', e,
#       args: [ '--optimizer', 'fAdaBelief', '--learningRate', '0.001', '--maxGenerations', '200' ],
#       timeout : 3600,
#       suite: 'regression',
#       workdir: meson.current_source_dir(),
#       depends: python_extension,
#       env: nomalloc
#     )
