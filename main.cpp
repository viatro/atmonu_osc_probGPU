#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <array>

using namespace std;

const double Pi = acos(-1.0);

class EarthCrossing {

  private:
    
    double zenith_angle;
    double production_height;
    double detector_depth;

    const size_t R = 6371000; // Earth radius in meters
    const size_t hAtm = 86000; // atmosphere height in meters
    map<long, double> earth_layers_density_map;
    vector<array<double, 3>> crossing_layers;

    void FillLayersDensities(size_t dr = 100) {
        map<int, array<double, 3> > earth_density_map;
        map<double, array<double, 3> > atmo_temperature_map;
        map<int, double> ye_map;

        ye_map[1221500] = 0.4656;
        ye_map[3480000] = 0.4656;
        ye_map[     R ] = 0.4957;
        ye_map[   2*R ] = 0.5   ;

        // r in meters, a[0] + a[1]*r + a[2]*r^2 density in kg/m^3
        earth_density_map[1221500] = {1.3088e4,  1.9110e-8, -2.1773e-10};
        earth_density_map[3480000] = {1.2346e4,  1.3976e-4, -2.4123e-10};
        earth_density_map[3630000] = {7.3067e3, -5.0007e-4,  0.        };
        earth_density_map[5701000] = {6.7823e3, -2.4441e-4, -3.0922e-11};
        earth_density_map[5771000] = {5.3197e3, -2.3286e-4,  0.        };
        earth_density_map[5971000] = {1.1249e4, -1.2603e-3,  0.        };
        earth_density_map[6151000] = {7.1083e3, -5.9706e-4,  0.        };
        earth_density_map[6346600] = {2.6910e3,  1.0869e-4,  0.        };
        earth_density_map[6356000] = {2.9000e3,  0.       ,  0.        };
        earth_density_map[6368000] = {2.6000e3,  0.       ,  0.        };
        earth_density_map[6371000] = {1.0200e3,  0.       ,  0.        };

        // geopotential h in meters, T in Kelvin, T lapse in K/m, density in g/cm^3
        atmo_temperature_map[    0.           ] = { 15.0  + 273.15, -0.0065, 1.2250e-3           };
        atmo_temperature_map[11000.           ] = {-56.5  + 273.15,  0.0   , 3.639271988833481e-4};
        atmo_temperature_map[20000.           ] = {-56.5  + 273.15,  0.001 , 8.80391839515545e-5 };
        atmo_temperature_map[32000.           ] = {-44.5  + 273.15,  0.0028, 1.3226067235598e-5  };
        atmo_temperature_map[47000.           ] = {- 2.5  + 273.15,  0.0   , 1.427313410448156e-6};
        atmo_temperature_map[51000.           ] = {- 2.5  + 273.15, -0.0028, 8.61479984576618e-7 };
        atmo_temperature_map[71000.           ] = {-58.5  + 273.15, -0.002 , 6.420472983251111e-8};
        atmo_temperature_map[84854.57642868205] = {-86.28 + 273.15,  0.0   , 6.954393506100748e-9};

        double rho = 0.0;
        for (size_t r = 0; r <= R + hAtm; r += dr) {
            if (r <= R) {
                array<double,3> pars = earth_density_map.lower_bound(r)->second;
                rho = 0.001*(pars[0] + pars[1]*r + pars[2]*r*r);
            } else if (r > R && r <= R + hAtm) {
                double h = r - R;
                h = h*R/(h + R); // convertion from geometric to geopotential

                auto iter = atmo_temperature_map.lower_bound(h);
                --iter;

                double g0 = 9.80665; // m/s^2
                double R = 8.3144598; // N·m/(mol·K)
                double M = 0.0289644; // kg/mol

                double hb   = iter->first;
                double Tb   = iter->second[0];
                double Lb   = iter->second[1];
                double rhob = iter->second[2];

                if (Lb == 0.0) {
                    rho = rhob * exp(-g0*M*(h - hb)/(R*Tb));
                } else {
                    rho = rhob * pow(Tb/(Tb + Lb*(h - hb)), 1 + g0*M/(R*Lb));
                }
            } else rho = 0.0;
            earth_layers_density_map[r] = rho;
        }
    }

    double GetYe(double r) {
        //if (r <= 1221500) return 0.4656;
        if (r <= 3480000) return 0.4656;
        if (r <= R      ) return 0.4957;
        else              return 0.5   ;
    }

    void FillCrossingLayers() {
        crossing_layers.clear();
        double rmin = R - detector_depth;
        double r = R + production_height;
        auto layer_iter = earth_layers_density_map.lower_bound(r);
        double rho = 0.0;
        double ye = 0.0;
        double length = 0.0;
        double rup = 0.0;
        if (zenith_angle == 0.0) {
            while (r > rmin) {
                rup = r;
                rho = layer_iter->second;
                ye = GetYe(r);
                --layer_iter;
                r = layer_iter->first;
                if (r < rmin) r = rmin;
                length = rup - r;
                crossing_layers.emplace_back(array<double, 3>{length, rho, ye});
            }
        } else if (zenith_angle == Pi) {
            rmin = 0.0;
            while (r > rmin) {
                rup = r;
                rho = layer_iter->second;
                ye = GetYe(r);
                --layer_iter;
                r = layer_iter->first;
                if (r < rmin) r = rmin;
                length = rup - r;
                crossing_layers.emplace_back(array<double, 3>{length, rho, ye});
            }
            double rmax = R - detector_depth;
            double rlow = rmin;
            while (r < rmax) {
                rlow = r;
                ++layer_iter;
                r = layer_iter->first;
                if (r > rmax) r = rmax;
                rho = layer_iter->second;
                ye = GetYe(r);
                length = r - rlow;
                crossing_layers.emplace_back(array<double, 3>{length, rho, ye});
            }
        } else {
            //crossing line y = k * x + b
            double k = cos(zenith_angle)/sin(zenith_angle);
            double b = R - detector_depth;
            //point of line with minimal distance to center of Earth, which is {x0,y0} = {0,0}
            double xmin = 0.0;
            double ymin = b;
            if (zenith_angle > Pi/2) {
                xmin = -k*b/(k*k + 1);
                ymin = xmin * k + b;
                rmin = hypot(xmin, ymin);
            }
            double D = -b*b + (1 + k*k)*r*r;
            double xlow = (-b*k + sqrt(D))/(1 + k*k);
            double ylow = xlow * k + b;
            double xup = 0.;
            double yup = 0.;

            while (r > rmin) {
                xup = xlow;
                yup = ylow;
                rup = r;
                rho = layer_iter->second;
                ye = GetYe(r);
                --layer_iter;
                r = layer_iter->first;
                if (r < rmin) r = rmin;
                D = -b*b + (1 + k*k)*r*r;
                xlow = (-b*k + sqrt(D))/(1 + k*k);
                ylow = xlow * k + b;
                length = hypot(xup - xlow, yup - ylow);
                crossing_layers.emplace_back(array<double, 3>{length, rho, ye});
            }

            if (zenith_angle > Pi/2) {
                double rmax = R - detector_depth;
                while (r < rmax) {
                    xup = xlow;
                    yup = ylow;
                    ++layer_iter;
                    r = layer_iter->first;
                    if (r > rmax) r = rmax;
                    rho = layer_iter->second;
                    ye = GetYe(r);
                    D = -b*b + (1 + k*k)*r*r;
                    xlow = (-b*k - sqrt(D))/(1 + k*k);
                    ylow = xlow * k + b;
                    length = hypot(xup - xlow, yup - ylow);
                    crossing_layers.emplace_back(array<double, 3>{length, rho, ye});
                }
            }
        }
    }

  public:

    EarthCrossing(double zenith_angle = 0.0, double production_height = 0.0, double detector_depth = 0.0) {
        FillLayersDensities(100);
        SetParams(zenith_angle, production_height, detector_depth);
    }

    ~EarthCrossing(){}

    void SetParams(double zenith_angle = 0.0, double production_height = 0.0, double detector_depth = 0.0) {
        this->zenith_angle = zenith_angle;
        this->production_height = production_height;
        this->detector_depth = detector_depth;

        FillCrossingLayers();
    }

    const vector<array<double, 3>>& GetCrossingLayers() const {
        return crossing_layers;
    }

};

extern /*"C"*/ void setMNS(float x12, float x13, float x23, float m21, float m23, float Delta = 0.0, bool kSquared = true);
extern /*"C"*/ void GetProb(int Alpha, int Beta, float Path, float Density, float ye, float *Energy, int n, float *oscw);

int main (int argc, char **argv) {

    setMNS(0.304, 0.0219, 0.514, 7.53e-5, 2.42e-3);

    int points_per_decade = 10000;

    vector<float> energies;
    for (long i = -6*points_per_decade; i <= 1*points_per_decade; ++i) {
        energies.push_back(pow10(i/double(points_per_decade)));
    }

    vector<vector<float>> osc_weights(3, vector<float>(energies.size(), 0.0));
    auto oscw_tmp = oscw_tmp_sum = osc_weights;


    EarthCrossing earth_crossing(Pi*0.75, 15000.);
    auto layers = earth_crossing.GetCrossingLayers();

    GetProb(1, 1, layers[0][0], layers[0][1], layers[0][2], energies.data(), energies.size(), osc_weights[0].data());
    GetProb(1, 2, layers[0][0], layers[0][1], layers[0][2], energies.data(), energies.size(), osc_weights[1].data());
    GetProb(1, 3, layers[0][0], layers[0][1], layers[0][2], energies.data(), energies.size(), osc_weights[2].data());

    for (size_t iLayer = 1; iLayer < layers.size(); ++iLayer) {
        for (int nuFlavorIn = 0; nuFlavorIn < 3; ++nuFlavorIn) {
            for (int nuFlavorOut = 0; nuFlavorOut < 3; ++nuFlavorOut) {
                GetProb(nuFlavorIn+1, nuFlavorOut+1, layers[0][0], layers[0][1], layers[0][2], energies.data(), energies.size(), oscw_tmp[nuFlavorOut].data());
                GetProb(nuFlavorIn+1, nuFlavorOut+1, layers[0][0], layers[0][1], layers[0][2], energies.data(), energies.size(), oscw_tmp[nuFlavorOut].data());
                GetProb(nuFlavorIn+1, nuFlavorOut+1, layers[0][0], layers[0][1], layers[0][2], energies.data(), energies.size(), oscw_tmp[nuFlavorOut].data());

                for (size_t i = 0; i < energies.size(); ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (oscw_tmp[j][i] < 0.0 || oscw_tmp[j][i] > 1.0 || oscw_tmp[j][i] != oscw_tmp[j][i]) {
                            cout << energies[i]*1000.0 << " MeV :\t" << j << "\t" << oscw_tmp[j][i] << endl;
                        }
                    }

                }
            }
        }
        cout << "\r" << iLayer << " / " << layers.size() << flush;

    }
    cout << endl;

    return 0;
}
