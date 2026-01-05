# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  

## [0.2.2] – 2026-01-05

### Added
- Docstrings for the `basis` module.
- Support for building cylindrical basis functions.
- Unit tests for the `basis` module updated to match refactored changes.
- Unit tests for `Profiles` were added.
- Support to compute EXP coefficients from particle positions and masses.
- MPI support for EXP coefficient computation.
- Unit tests for EXP coefficient computation.
- Unit tests performed on a Milky Way simulated galaxy.
- Matplotlib `exptools` style implemented in `utils/`.
- Unit tests for the plotting style.

### Changed
- Renamed `basis_build` folder to match pyEXP naming conventions.
- `basis` module refactored.
- Moved `profiles.py` from `basis` to `utils/`.

### Fixed
- Fixed a typo in `make_model`.
- Prevented `log10` of zero in density smoothing.

