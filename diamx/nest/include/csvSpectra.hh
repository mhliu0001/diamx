// csvSpectra.hh: Sample energy from csv spectrum file
// Code generated with ChatGPT-4o
// Modified to use NEST's random generator

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include "RandomGen.hh"

class EnergySampler {
public:
    // Constructor to load data from CSV file and seed the random generator
    EnergySampler(const std::string& filename) {
        loadDataFromCSV(filename);
        ensureSorted();
        removeDuplicates();
        normalizePDF();
        computeCumulativeDistribution();
    }

    // Function to sample an energy value from the interpolated PDF
    double sampleEnergy() {
        double u = RandomGen::rndm()->rand_uniform();  // Random number [0,1]
        return interpolateEnergyFromPDF(u);
    }

private:
    std::vector<double> energies;
    std::vector<double> pdf_values;
    std::vector<double> cdf_values;
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};

    // Load data from CSV file
    void loadDataFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file");
        }

        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string energy_str, pdf_str;

            std::getline(ss, energy_str, ',');
            std::getline(ss, pdf_str, ',');

            double energy = std::stod(energy_str);
            double pdf_value = std::stod(pdf_str);

            if (pdf_value < 0) {
                throw std::runtime_error("Invalid PDF value: PDF values must be non-negative.");
            }

            energies.push_back(energy);
            pdf_values.push_back(pdf_value);
        }

        if (energies.empty() || pdf_values.empty()) {
            throw std::runtime_error("CSV file contains no valid data.");
        }

        file.close();
    }

    // Ensure the energies are sorted in ascending order
    void ensureSorted() {
        std::vector<std::pair<double, double>> energy_pdf_pairs;
        for (size_t i = 0; i < energies.size(); ++i) {
            energy_pdf_pairs.emplace_back(energies[i], pdf_values[i]);
        }

        std::sort(energy_pdf_pairs.begin(), energy_pdf_pairs.end());

        for (size_t i = 0; i < energy_pdf_pairs.size(); ++i) {
            energies[i] = energy_pdf_pairs[i].first;
            pdf_values[i] = energy_pdf_pairs[i].second;
        }
    }

    // Remove duplicate energy values, averaging their PDFs
    void removeDuplicates() {
        std::vector<double> new_energies;
        std::vector<double> new_pdf_values;

        for (size_t i = 0; i < energies.size(); ++i) {
            if (i == 0 || energies[i] != energies[i - 1]) {
                new_energies.push_back(energies[i]);
                new_pdf_values.push_back(pdf_values[i]);
            } else {
                // Average the PDF values for duplicate energies
                new_pdf_values.back() = (new_pdf_values.back() + pdf_values[i]) / 2.0;
            }
        }

        energies = std::move(new_energies);
        pdf_values = std::move(new_pdf_values);
    }

    // Normalize the PDF to ensure its integral equals 1
    void normalizePDF() {
        // Calculate the integral of the PDF using trapezoidal rule
        double integral = 0.0;
        for (size_t i = 1; i < energies.size(); ++i) {
            double dx = energies[i] - energies[i - 1];
            double avg_pdf = (pdf_values[i] + pdf_values[i - 1]) / 2.0;
            integral += avg_pdf * dx;
        }

        // Normalize the PDF by dividing by the integral
        if (integral <= 0.0) {
            throw std::runtime_error("The PDF cannot be normalized due to zero or negative integral.");
        }

        for (auto& value : pdf_values) {
            value /= integral;
        }
    }

    // Compute cumulative distribution function (CDF) from normalized PDF
    void computeCumulativeDistribution() {
        cdf_values.resize(pdf_values.size());
        cdf_values[0] = 0.0;

        // Trapezoidal rule to compute the cumulative sum (CDF)
        for (size_t i = 1; i < pdf_values.size(); ++i) {
            double dx = energies[i] - energies[i - 1];
            double avg_pdf = (pdf_values[i] + pdf_values[i - 1]) / 2.0;
            cdf_values[i] = cdf_values[i - 1] + avg_pdf * dx;
        }

        // Normalize CDF to ensure it ends at 1
        double cdf_max = cdf_values.back();
        for (auto& value : cdf_values) {
            value /= cdf_max;
        }
    }

    // Interpolate the energy from the PDF
    double interpolateEnergyFromPDF(double u) const {
        // Find the corresponding segment in the CDF
        auto it = std::lower_bound(cdf_values.begin(), cdf_values.end(), u);

        if (it == cdf_values.begin()) {
            return energies.front();
        } else if (it == cdf_values.end()) {
            return energies.back();
        }

        size_t idx = std::distance(cdf_values.begin(), it);

        // Linear interpolation on the PDF between points
        double e1 = energies[idx - 1];
        double e2 = energies[idx];
        double p1 = pdf_values[idx - 1];
        double p2 = pdf_values[idx];
        double cdf1 = cdf_values[idx - 1];
        double cdf2 = cdf_values[idx];

        // Linearly interpolate energy between the two points in the CDF range
        return e1 + (u - cdf1) * (e2 - e1) / (cdf2 - cdf1);
    }
};