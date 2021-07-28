#pragma once

#include <simdjson.h>
#include <glm/glm.hpp>
#include <optional>

namespace RLpbr {
namespace editor {

class JSONReader {
public:
    JSONReader(const JSONReader &) = delete;
    JSONReader(JSONReader &&) = default;

    inline simdjson::ondemand::document &getDocument() { return doc_; }

    static std::optional<JSONReader> loadGZipped(const char *filename,
        simdjson::ondemand::parser &&parser);

    static std::optional<glm::vec3> parseVec3(
        simdjson::simdjson_result<simdjson::ondemand::value> obj,
        const char *field);
 
    static simdjson::ondemand::parser && takeParser(JSONReader &&json);
private:
    JSONReader(simdjson::ondemand::parser &&parser);

    simdjson::ondemand::parser parser_;
    std::vector<uint8_t> data_;
    simdjson::ondemand::document doc_;
};

}
}
