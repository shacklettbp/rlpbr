#include "json.hpp"

#include <zlib.h>

using namespace std;
using namespace simdjson;
using namespace simdjson::ondemand;

namespace RLpbr {
namespace editor {

optional<JSONReader> JSONReader::loadGZipped(const char *filename,
                                             parser &&parser)
{
    constexpr int init_size = 4096;

    gzFile gz = gzopen(filename, "rb");
    if (gz == nullptr) {
        cerr << "Failed to open " << filename << endl;
        return optional<JSONReader>();
    }

    auto json = make_optional<JSONReader>(move(parser));
    json->data_.resize(init_size);

    size_t num_bytes_decompressed = 0;

    while (!gzeof(gz)) { 
        int bytes_read =
            gzread(gz, json->data_.data() + num_bytes_decompressed,
                   json->data_.size() - num_bytes_decompressed);
        if (bytes_read <= 0) {
            break;
        }
        num_bytes_decompressed += bytes_read;

        json->data_.resize(json->data_.size() * 2);
    }

    if (!gzeof(gz)) {
        int zlib_err;
        gzerror(gz, &zlib_err);
        if (zlib_err != Z_STREAM_END) {
            cerr << "Failed to read " << filename << endl;
            return optional<JSONReader>();
        }
    }
    gzclose(gz);

    json->data_.resize(num_bytes_decompressed + simdjson::SIMDJSON_PADDING);
    auto result = parser.iterate(json->data_.data(), num_bytes_decompressed,
                                 json->data_.size());

    if (result.error()) {
        cout << "Failed to load: " << result.error() << endl;

        return optional<JSONReader>();
    }

    json->doc_ = move(result).value_unsafe();

    return json;
}

JSONReader::JSONReader(parser &&parser)
    : parser_(move(parser)),
      data_(),
      doc_()
{}

parser && JSONReader::takeParser(JSONReader &&json)
{
    return move(json.parser_);
}

optional<glm::vec3> JSONReader::parseVec3(simdjson_result<value> obj, const char *field)
{
    glm::vec3 result;
    int idx = 0;
    for (auto component : obj[field]) {
        double v;
        auto err = component.get(v);
        if (err) {
            return optional<glm::vec3>();
        }
        result[idx++] = v;

        if (idx >= 3) {
            break;
        }
    }

    if (idx == 3) {
        return result;
    } else {
        return optional<glm::vec3>();
    }
}

}
}
