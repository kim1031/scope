#include <cstdint>
#include <iostream>
#include <fstream>
#include <csignal>
#include <vector>
#include <tuple>
#include <chrono>
#include <unistd.h>
#include <energymon/energymon.h>
#include <energymon/energymon-default.h>

using namespace std;

char SHOULD_EXIT = 0;
char REINIT_ENERGYMON = 0;

typedef tuple<uint64_t, uint64_t, double> energy_record;

void sigchld_handler(int sig) { SHOULD_EXIT = 1; }
void sigusr_handler(int sig) { REINIT_ENERGYMON = 1; }

vector<energy_record> *read_values(string fname, long sleep_time, long em_life)
{
    // vector<energy_record> *records = new vector<energy_record>();
    // records->reserve(2000);
    ofstream outfile(fname, ios::out);
    outfile << "time,energy\n";

    energymon em;
    energymon_get_default(&em);
    em.finit(&em);

    uint64_t start_uj = em.fread(&em);
    uint64_t prev_uj = start_uj;
    auto start_time = chrono::steady_clock::now();
    auto prev_time = start_time;
    while (!SHOULD_EXIT) {
        auto curr_time = chrono::steady_clock::now();

        uint64_t time_diff =
            (uint64_t)chrono::duration_cast<chrono::milliseconds>(
                curr_time - start_time).count(); 
        uint64_t current_uj = em.fread(&em);

        if (REINIT_ENERGYMON || current_uj == 0) {
            em.ffinish(&em);
            energymon_get_default(&em);
            em.finit(&em);
            current_uj = prev_uj = start_uj = em.fread(&em);
            REINIT_ENERGYMON = 0;
        }

        uint64_t total_uj = current_uj - start_uj;
        
        uint64_t prev_time_diff = 
            (uint64_t)chrono::duration_cast<chrono::milliseconds>(
                curr_time - prev_time).count();
        uint64_t energy_delta = current_uj - prev_uj;
        
        prev_uj = current_uj;
        prev_time = curr_time;
        
        double power_consumption = ((double)energy_delta / 1000000.0) / ((double)prev_time_diff / 1000.0);

        // records->push_back(make_tuple(time_diff, total_uj, power_consumption));
        outfile << time_diff << "," << total_uj << "," << power_consumption << endl;
        outfile.flush();
        usleep(sleep_time);
    }

    em.ffinish(&em);
    outfile.close();
    // return records;
}

void write_to_file(string fname, vector<energy_record> *records)
{
    ofstream outfile(fname, ios::out);
    outfile << "time,energy\n";
    for (auto const& [time, energy, power] : *records) {
        outfile << time << "," << energy << "," << power << endl;
    }
    outfile.close();
}

int main(int argc, char **argv)
{
    unsigned int millis;
    unsigned int reinit_time;
    if (argc < 2) {
        millis = 1000;
    } else {
        sscanf(argv[1], "%du", &millis);
        //sscanf(argv[2], "%du", &reinit_time);
    }
    long sleep_time = (long)millis * 1000;
    long em_life = 12000;

    signal(SIGINT, sigchld_handler);
    signal(SIGUSR1, sigusr_handler);

    read_values(argv[2], sleep_time, em_life);
    // auto records = read_values(sleep_time);

    // write_to_file(argv[2], records);

    // delete records;
}