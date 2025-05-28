#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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


    m.def("lz_model", &LZModel,
          "Return the s1c_phd and s2c_phd with LZ model.\n"
          "Parameters:\n"
          "  detector_name (str): Specifies which LZ detector to use. Can be 'lz_ws2022', 'lz_sr1' or 'lz_ws2024'.\n"
          "  type (str): Interaction type. Supported types are 'NR', 'ER', 'DEC', 'beta'.\n"
          "  seed (uint64): Seed. If 0, use current time.\n"
          "  spectrumFileName (str): Spectrum to be used in energy sampling. Must be a csv file.\n"
          "  numEvts (uint64): Number of simulated events.\n"
          "Returns:\n"
          "  InferenceObservableArray: class (not dict) with two attributes:\n"
          "    s1c_phd (list): corrected S1 in LZ phd units.\n"
          "    s2c_phd (list): corrected S2 in LZ phd units.\n"
    );
}