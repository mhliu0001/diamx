#include "nestDiamxCore.hh"

int main() {
    int batch_size = 1000000;
    InferenceObservableArray result = LZModel(
        "lz_ws2024",
        "NR",
        0,
        "/home/mhliu/Documents/diamx/axion_med_1000GeV.csv",
        batch_size
    );
    assert(result.s1c_phd.size() == batch_size);
    assert(result.s2c_phd.size() == batch_size);
    cout << "Test completed with no errors." << endl;
    return 0;
}