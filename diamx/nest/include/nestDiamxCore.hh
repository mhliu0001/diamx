#include "NEST.hh"
// #include "analysis.hh"
#include "csvSpectra.hh"
#include <iostream>
#include <time.h>
#include <filesystem>

using namespace std;

struct InferenceObservableArray {
    vector<double> s1c_phd;
    vector<double> s2c_phd;
    vector<double> energy_rec;
};

// vector<double> NRYieldsParam;
// vector<double> ERYieldsParam;
// vector<double> NRERWidthsParam;

NEST::INTERACTION_TYPE getNESTType(string type);

// A simplified function based on execNEST, with much less functionality
// and error detection, but serves our purpose.
InferenceObservableArray LZModel(
    string detector_name,
    string type,
    uint64_t seed,
    string spectrumFileName,
    uint64_t numEvts,
    double dec_quenching_factor=1.0
);