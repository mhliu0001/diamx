#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "nestDiamxCore.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(nestDiamx, m) {
    m.doc() = "NEST diamx python binding";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<InferenceObservableArray>(m, "InferenceObservableArray")
        .def_readwrite("s1c_phd", &InferenceObservableArray::s1c_phd)
        .def_readwrite("s2c_phd", &InferenceObservableArray::s2c_phd)
        .def_readwrite("energy_rec", &InferenceObservableArray::energy_rec)
        .def("__repr__", [](const InferenceObservableArray &a) {
            // Shows all elements if the size is < 20; otherwise, shows first 5 and last 5 elements/
            std::ostringstream represent;

            auto appendArray = [&represent](const std::string &name, const std::vector<double> &values) {
                represent << "\"" << name << "\": [";

                size_t size = values.size();
                if (size > 20) {
                    // Add first 5 elements
                    for (size_t i = 0; i < 5; ++i) {
                        represent << values[i] << ", ";
                    }
                    represent << "...";
                    // Add last 5 elements
                    for (size_t i = size - 5; i < size; ++i) {
                        represent << ", " << values[i];
                    }
                } else {
                    // Add all elements if size is <= 10
                    for (size_t i = 0; i < size; ++i) {
                        represent << ", " << values[i];
                    }
                }

                represent << "]";
            };

            represent << "{";
            appendArray("s1c_phd", a.s1c_phd);
            represent << ", ";
            appendArray("s2c_phd", a.s2c_phd);
            represent << ", ";
            appendArray("energy_rec", a.energy_rec);
            represent << "}";

            return represent.str();
        });


    m.def(
        "lz_model",
        [](const std::string& detector_name,
           const std::string& type,
           std::uint64_t seed,
           const std::string& spectrumFileName,
           std::uint64_t numEvts,
           double dec_quenching_factor,
           bool high_energy) {
            if (high_energy) {
                // In high-energy mode the NEST library prints an "energy beyond the
                // AmBe endpoint of about 300 keV" warning to std::cerr for every
                // out-of-range event, which floods the output. Suppress it (only
                // when high_energy is explicitly requested) by redirecting std::cerr
                // to a discarded in-memory buffer for the duration of the call.
                auto buffer = py::module_::import("io").attr("StringIO")();
                py::scoped_ostream_redirect redirect_cerr(std::cerr, buffer);
                return LZModel(detector_name, type, seed, spectrumFileName,
                               numEvts, dec_quenching_factor, high_energy);
            }
            return LZModel(detector_name, type, seed, spectrumFileName,
                           numEvts, dec_quenching_factor, high_energy);
        },
        py::arg("detector_name"),
        py::arg("type"),
        py::arg("seed"),
        py::arg("spectrumFileName"),
        py::arg("numEvts"),
        py::arg("dec_quenching_factor") = 1.0,
        py::arg("high_energy") = false,
        "Return the s1c_phd and s2c_phd with LZ model.\n"
        "Parameters:\n"
        "  detector_name (str): Specifies which LZ detector to use. Can be 'lz_ws2022', 'lz_sr1' or 'lz_ws2024'.\n"
        "  type (str): Interaction type. Supported types are 'NR', 'ER', 'DEC', 'beta'.\n"
        "  seed (uint64): Seed. If 0, use current time.\n"
        "  spectrumFileName (str): Spectrum to be used in energy sampling. Must be a csv file.\n"
        "  numEvts (uint64): Number of simulated events.\n"
        "  dec_quenching_factor (float): Quenching factor for decay (DEC/EC) events. Default 1.0.\n"
        "  high_energy (bool): If True, extend the S1/S2 acceptance beyond the NEST\n"
        "    validation range and silence the AmBe-endpoint warnings, so high-energy\n"
        "    NR (e.g. exothermic spectra past ~300 keV) is simulated instead of clipped.\n"
        "    Use only when you only need in/out-of-ROI classification. Default False.\n"
        "Returns:\n"
        "  InferenceObservableArray: class (not dict) with two attributes:\n"
        "    s1c_phd (list): corrected S1 in LZ phd units.\n"
        "    s2c_phd (list): corrected S2 in LZ phd units.\n"
    );
}