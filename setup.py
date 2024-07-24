import setuptools

setuptools.setup(
        name="EXP_tools",
        version="0.1",
        author="B-BFE collaboration",
        author_email="ngaravito@flatironinstitute.org",
        description="Analysis tools for EXP and other BFE packages",
        packages=["EXPtools", "EXPtools/basis_builder/", "EXPtools/visuals", "EXPtools/utils", "EXPtools/scf"],
	install_requires=["healpy", "numpy","scipy"]
        )
