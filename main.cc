#include <iostream>
#include <cmath>
#include <vector>

extern "C" void setMNS(double x12, double x13, double x23, double m21, double m23, double Delta = 0.0, bool kSquared = true);
extern "C" void GetProb(int Alpha, int Beta, double Path, double Density, double ye, double *Energy, int n, double *oscw);

int main (int argc, char **argv) {

    setMNS(0.304, 0.0219, 0.514, 7.53e-5, 2.42e-3);

    int points_per_decade = 100000;

    std::vector<double> energies;
    for (long i = -6*points_per_decade; i <= 1*points_per_decade; ++i) {
        energies.push_back(pow10(i/double(points_per_decade)));
    }

    std::vector<std::vector<double>> osc_weights(4, std::vector<double>(energies.size(), 0.0));

    GetProb(2, 1, 2.*6371., 13.0, 0.5, &energies[0], energies.size(), &osc_weights[1][0]);
    GetProb(2, 2, 2.*6371., 13.0, 0.5, &energies[0], energies.size(), &osc_weights[2][0]);
    GetProb(2, 3, 2.*6371., 13.0, 0.5, &energies[0], energies.size(), &osc_weights[3][0]);

    for (size_t i = 0; i < energies.size(); ++i) {
        for (int j = 1; j <= 3; ++j) {
            //if (osc_weights[j][i] < 0.0 || osc_weights[j][i] > 1.0) {
                std::cout << energies[i]*1000.0 << " MeV :\t" << j << "\t" << osc_weights[j][i] << std::endl;
            //}
        }

    }

    return 0;
}
