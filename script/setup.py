from setuptools import setup, Extension

setup(
    name='my_module',
    ext_modules=[
        Extension(
            'my_module',
            sources=['nonsingularPredict/src'],
            include_dirs=['nonsingularPredict/include'],
            library_dirs=['nonsingularPredict/build'],
            libraries=['dace']
        )
    ]
)
