#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std; 

class ConfigParser {

private:
    unordered_map<string, string> settings;

public:
    ConfigParser(const string& path) {

        ifstream file(path);
        string line;

        if (!file) {
            cout << "Error opening config file!" << endl;
        }

        while (getline(file, line)) {
            istringstream is_line(line);
            string key;

            if (getline(is_line, key, '=')) {
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


class Config {

private:
ConfigParser parser;

public: 

const int test1;
const float test2;
const string test3;

Config(const string &path) : parser(path), 
    test1(parser.get<int>("test1", 0)),
    test2(parser.get<float>("test2", 0.0)),
    test3(parser.get<string>("test3", ":(")) {}

};
