#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "../util.hpp"

using namespace std; 

std::istream& operator>>(std::istream& is, vec3F& v) {
    is >> v.x >> v.y >> v.z;
    if (!is) {
        is.clear();
    }
    return is;
}

std::istream& operator>>(std::istream& is, Fractal& v) {
    std::string name;
    is >> name;

    if (name == "mandelbox") {
        v = MANDELBOX;
    } else if (name == "mandelbulb") {
        v = MANDELBULB;
    } else if (name == "mengerprison") {
        v = MENGER_PRISON;
    } else {
        v = MENGER_SPONGE;
    } 

    if (!is) {
        is.clear();
    }
    return is;
}
class ConfigParser {

private:
    unordered_map<string, string> settings;

public:
    ConfigParser(const string& path) {

        ifstream file(path);
        string line;

        if (!file) {
            cout << "Error opening config file! Path: " << path << endl;
        }

        while (getline(file, line)) {
            istringstream is_line(line);
            string key;
            
            if (getline(is_line, key, '=')) {
                if (key.at(0) == '#')
                    continue; // comment
                
                string value;
                if (getline(is_line, value)) {
                    settings[key] = value;
                }
            }
        }

        file.close();
    }

    template<typename T>
    T get(const string& key, T default_value) const {

        try {
            istringstream iss(settings.at(key));
            T result;
            iss >> result;
            return result;

        } catch (out_of_range) {
            return default_value;
        }
    }
};


struct Config {

public: 

const int iterations;
const Fractal fractal;
const float mandelbulb_power;
const float mandelbox_mu;
const float mandelbox_mR;
const float mandelbox_fR;

const vec3F backgroundColor;

// lighting
const vec3F ambientColor;
const vec3F lightDirection;
const vec3F lightColor;
const vec3F materialColor;
const float directionalLightStrength;
const float cameraLightStrength;
const int ambientOcclusionLevels;
const float specularLightStrength;
const int specularLightFocus;
const float fogDistance;

// fractal quality
const int maxMarchingSteps;
const int maxShadowSteps;
const int antiAliasing;

    Config(const ConfigParser& parser) : 
        iterations(parser.get<int>("iterations", 5)),
        fractal(parser.get<Fractal>("fractal", MENGER_SPONGE)),
        mandelbulb_power(parser.get<float>("mandelbulb_power", 8.0)),
        mandelbox_mu(parser.get<float>("mandelbox_mu", 1)),
        mandelbox_mR(parser.get<float>("mandelbox_mR", 1)),
        mandelbox_fR(parser.get<float>("mandelbox_fR", 1)),
        backgroundColor(parser.get<vec3F>("backgroundColor", {0.0, 0.0, 0.0})),
        ambientColor(parser.get<vec3F>("ambientColor", {50.0, 100.0, 40.0})),
        lightDirection(parser.get<vec3F>("lightDirection", {-0.4, 0.2, 1.0})),
        lightColor(parser.get<vec3F>("lightColor", {120.0, 40.0, 136.0})),
        materialColor(parser.get<vec3F>("materialColor", {0.0, 0.0, 0.0})),
        cameraLightStrength(parser.get<float>("cameraLightStrength", 0.0)),
        directionalLightStrength(parser.get<float>("directionalLightStrength", 1.0)),
        ambientOcclusionLevels(parser.get<int>("ambientOcclusionLevels", 32)),
        specularLightStrength(parser.get<float>("specularLightStrength", 1.0)),
        specularLightFocus(parser.get<int>("specularLightFocus", 32)),
        fogDistance(parser.get<float>("fogDistance", 8.0)),
        maxMarchingSteps(parser.get<int>("maxMarchingSteps", 512)),
        maxShadowSteps(parser.get<int>("maxShadowSteps", 150)),
        antiAliasing(parser.get<int>("antiAliasing", 0)) {}
};
