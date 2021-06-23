from setuptools import setup
exec(open('SeqIO/version.py').read())

setup(
    name='seq-io',
    version=__version__,
    description="Import .seq files into hyperspy",
    author='Carter Francis',
    author_email='csfrancis@wisc.edu',
    license='MIT',
    url="https://github.com/CSSFrancis/SeqIO",
    keywords=[
        "data analysis",
        "diffraction",
        "microscopy",
        "electron diffraction",
        "electron microscopy",
    ],
    install_requires=[
        "scikit-image >= 0.17.1",
        "matplotlib >= 3.1.1",  # 3.1.0 failed
        "scikit-learn >= 0.19",  # reason unknown
        "hyperspy >= 1.6.2",  # earlier versions incompatible with numpy >= 1.17.0 and hyperspy == 1.6.0 has a histogram bug
        "diffsims >= 0.3",  # Makes use of functionality introduced in this release
        "lmfit >= 0.9.12",
        "numpy>= 1.20.0",
        "pyfai",
        "ipywidgets",
        "numba",
        "orix >= 0.3",
        "pyxem ",
        "xmltodict"
    ],
    packages=['SeqIO'],
)
