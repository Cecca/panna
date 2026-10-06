#include <highfive/H5Easy.hpp>

#include <chrono>
#include <filesystem>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/rand.hpp"
#include "panna/trieindex.hpp"

float compute_recall( const std::vector<std::pair<float, uint32_t>>& ground,
                      const std::vector<std::pair<float, uint32_t>>& actual ) {
    size_t k = actual.size();
    float thresh = ground[k - 1].first;

    float cnt = 0.0;
    for ( auto pair : actual ) {
        if ( pair.first <= thresh ) {
            cnt += 1.0;
        }
    }

    return cnt / k;
}

int main( int , char** ) {
    using Distance = panna::EuclideanDistance;
    using Dataset = panna::EuclideanPoints;
    using HasherBuilder = panna::E2LSHBuilder<8, Dataset>;
    using Hasher = HasherBuilder::Output;

    H5Easy::File file( "fashion-mnist-784-euclidean.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> data =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );

    std::vector<std::vector<float>> queries = H5Easy::load<std::vector<std::vector<float>>>( file, "/test" );
    queries.resize( 1000 ); // keep only some queries
    // std::vector<std::vector<float>> queries;
    // queries.push_back(H5Easy::load<std::vector<std::vector<float>>>( file, "/test" )[896]);
    std::cout << "data loaded" << std::endl;

    size_t dimensions = data[0].size();
    HasherBuilder hbuilder( dimensions );

    panna::Index<Dataset, Hasher, Distance> index( dimensions, hbuilder, 32 );
    for ( auto& p : data ) {
        index.insert( p.begin(), p.end() );
    }
    index.rebuild();
    std::cerr << "index ready" << std::endl;

    size_t k = 10;
    float delta = 0.2;
    std::vector<std::vector<std::pair<float, uint32_t>>> res;
    std::vector<std::vector<std::pair<float, uint32_t>>> res_prob;
    res.resize( queries.size() );
    res_prob.resize( queries.size() );

    std::cout << "compute ground truth" << std::endl;
    size_t q_idx = 0;
    for ( auto q : queries ) {
        index.search_brute_force( q.begin(), q.end(), k, res[q_idx++] );
    }

    std::cout << "run queries" << std::endl;
    q_idx = 0;
    auto start = std::chrono::steady_clock::now();
    for ( auto q : queries ) {
        index.search( q.begin(), q.end(), k, delta, res_prob[q_idx++] );
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_s =
        std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() /
        ( 1000.0 * 1000.0 );
    double qps = res.size() / elapsed_s;
    float avg_recall = 0.0;
    for ( size_t q_idx = 0; q_idx < res.size(); q_idx++ ) {
        float rec = compute_recall( res.at( q_idx ), res_prob.at( q_idx ) );
        avg_recall += rec;
        // if ( rec < 0.4 ) {
        //     dbg(q_idx);
        //     std::vector<float> q = queries[q_idx];
        //     std::vector<std::pair<float, uint32_t>> check_exact;
        //     std::vector<std::pair<float, uint32_t>> check_prob;
        //     index.search_brute_force( q, k, check_exact );
        //     index.search( q, k, delta, check_prob );
        //     std::sort(check_exact.begin(), check_exact.end());
        //     std::sort(check_prob.begin(), check_prob.end());
        //     dbg(compute_recall(check_exact, check_prob));
        //     dbg(check_exact);
        //     dbg(check_prob);
        // }
    }
    avg_recall /= res.size();
    std::cout << res.size() << " queries " << "qps: " << qps << " average rcall " << avg_recall
              << " expected recall " << 1 - delta
              << " total number of collisions: " << panna::g_collisions
              << std::endl;
}
