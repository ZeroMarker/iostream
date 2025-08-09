//
// Created by ttft3 on 2025/8/10.
//

// magnet_downloader.cpp
// Build: g++ -std=c++17 magnet_downloader.cpp -o magnet_downloader $(pkg-config --cflags --libs libtorrent-rasterbar) -pthread

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>

#include <libtorrent/session.hpp>
#include <libtorrent/settings_pack.hpp>
#include <libtorrent/add_torrent_params.hpp>
#include <libtorrent/magnet_uri.hpp>
#include <libtorrent/torrent_handle.hpp>
#include <libtorrent/torrent_status.hpp>
#include <libtorrent/alert_types.hpp>
#include <libtorrent/alert.hpp>
#include <libtorrent/alert_factory.hpp>

namespace lt = libtorrent;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " \"<magnet-link>\" <save_path>\n";
        return 1;
    }

    std::string magnet = argv[1];
    std::string save_path = argv[2];

    try {
        // 1) session settings
        lt::settings_pack settings;
        // get all alert categories so we can see status/errors
        settings.set_int(lt::settings_pack::alert_mask, lt::alert::all_categories);
        // listen on default bittorrent ports on all interfaces
        settings.set_str(lt::settings_pack::listen_interfaces, "0.0.0.0:6881");

        lt::session ses(settings);

        // (optional) enable DHT (should be enabled by default in many builds)
        // ses.add_dht_router({ "router.utorrent.com", 6881 });

        // 2) parse magnet and fill add_torrent_params
        lt::add_torrent_params params = lt::parse_magnet_uri(magnet);
        params.save_path = save_path;
        // ensure auto-managed and not paused (adjust flags if you want different behavior)
        params.flags |= lt::torrent_flags::auto_managed;
        params.flags &= ~lt::torrent_flags::paused;

        // 3) add torrent
        lt::torrent_handle th = ses.add_torrent(std::move(params));
        if (!th.is_valid()) {
            std::cerr << "Failed to add torrent\n";
            return 2;
        }

        // 4) main loop: print progress and process alerts
        while (true) {
            // pop all alerts and print them
            std::vector<lt::alert*> alerts;
            ses.pop_alerts(&alerts);
            for (lt::alert* a : alerts) {
                // print short description for debugging
                std::cout << "[" << a->what() << "] " << a->message() << "\n";
            }

            // get torrent status (may be empty right away while metadata is being fetched)
            lt::torrent_status st = th.status(lt::torrent_handle::query_save_path
                                             | lt::torrent_handle::query_state
                                             | lt::torrent_handle::query_peers
                                             | lt::torrent_handle::query_name
                                             | lt::torrent_handle::query_progress
                                             | lt::torrent_handle::query_seconds_download);

            double progress = st.progress * 100.0;
            int peers = st.num_peers;
            int seeds = st.num_seeds;
            std::int64_t dl_rate = st.download_rate; // bytes/sec
            std::int64_t ul_rate = st.upload_rate;   // bytes/sec
            std::string state = lt::torrent_status::state_t(st.state).to_string(); // not always available on all builds

            // Friendly print (human readable rates)
            auto human = [](std::int64_t bps){
                double v = double(bps);
                const char* suf[] = {"B/s","KB/s","MB/s","GB/s"};
                int i = 0;
                while (v >= 1024.0 && i < 3) { v/=1024.0; ++i; }
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.2f %s", v, suf[i]);
                return std::string(buf);
            };

            std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << progress << "% "
                      << "| Peers: " << peers << " (seeds: " << seeds << ") "
                      << "| DL: " << human(dl_rate) << " UL: " << human(ul_rate)
                      << " | Name: " << st.name << "           " << std::flush;

            // finished ?
            if (st.is_seeding) {
                std::cout << "\nDownload complete (seeding). Saved to: " << st.save_path << "\n";
                break;
            }

            // sleep briefly
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // optionally, request save_resume_data and gracefully shutdown
        if (th.is_valid()) {
            th.save_resume_data();
            // give it a second to produce the resume alert
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::vector<lt::alert*> alerts;
            ses.pop_alerts(&alerts);
            for (lt::alert* a : alerts) {
                std::cout << "[" << a->what() << "] " << a->message() << "\n";
            }
        }
    }
    catch (std::exception& ex) {
        std::cerr << "\nException: " << ex.what() << "\n";
        return 3;
    }

    return 0;
}
