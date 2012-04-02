/*
 * util.c
 *
 *  Created on: Nov 16, 2009
 *      Author: timon
 */

#include "util.h"

vector<void*> MemoryManagement::ptrs;
vector<unsigned> MemoryManagement::sizes;

FILE* Util::openFile(string path)
{
    FILE* dataFile;
    if (!(dataFile = fopen(path.data(), "w"))) {
        string error = "Error opening " + path;
        throw error;
    }
    //	printf(" opening file \"%s\"\n", path.data());
    return dataFile;
}

void Util::check(bool condition, std::string error)
{
    if (condition){
        throw error;
    }
}

void* MemoryManagement::malloc(unsigned size)
{
    if (size == 0) {
        std::string error = "Cannot allocate 0 bytes";
        throw error;
    }
    void* toReturn = std::malloc(size);

    ptrs.push_back(toReturn);
    sizes.push_back(size);

    return toReturn;
}

void MemoryManagement::free(void* ptr)
{
    char found = 0;
    for (int i = 0; i < ptrs.size(); i++) {
        if (ptrs[i] == ptr) {
            ptrs.erase(ptrs.begin() + i);
            sizes.erase(sizes.begin() + i);
            std::free(ptr);
            found = 1;
            break;
        }
    }
    if (!found) {
        // TODO pensarse lo de fprintf(stderr, ) y por lo menos hacerlo en todos sitios igual
        cout << "Unable to free " << (unsigned)ptr << endl;
        std::string error = "The pointer to free wasn't found";
        throw error;
    }
}

void MemoryManagement::clear()
{
    for (int i = 0; i < ptrs.size(); i++) {
        std::free(ptrs[i]);
    }
    ptrs.clear();
    sizes.clear();
}

void MemoryManagement::printTotalAllocated()
{
    unsigned totalAllocated = 0;
    for (int i = 0; i < sizes.size(); i++) {
        totalAllocated += sizes[i];
    }
    unsigned mb, kb, b;
    kb = totalAllocated / 1024;
    b = totalAllocated % 1024;
    if (kb == 0) {
        mb = 0;
    } else {
        mb = kb / 1024;
        kb = kb % 1024;
    }
    cout << "There are " << mb << " MB " << kb << " KB and " << b << " Bytes allocated. ( total "
            << totalAllocated << " Bytes )" << endl;
}

void MemoryManagement::printTotalPointers()
{
    cout << "There are " << ptrs.size() << " pointers allocated." << endl;
}

void MemoryManagement::printListOfPointers()
{
    for (int i = 0; i < ptrs.size(); i++) {
        printf(" %d mem_address %d  size = %d \n", i, (unsigned)ptrs[i], sizes[i]);
    }
}

unsigned MemoryManagement::getPtrCounter()
{
    return ptrs.size();
}

unsigned MemoryManagement::getTotalAllocated()
{
    unsigned totalAllocated = 0;
    for (int i = 0; i < sizes.size(); i++) {
        totalAllocated += sizes[i];
    }
}

int Random::integer(unsigned range)
{
    return (rand() % (2 * range + 1)) - range;
}

float Random::floatNum(float range)
{
    return ((rand() / (float)RAND_MAX) * (2 * range)) - range;
}

unsigned Random::positiveInteger(unsigned range)
{
    return rand() % range;
}

float Random::positiveFloat(float range)
{
    return (rand() / (float)RAND_MAX) * (range);
}

SimpleGraph::~SimpleGraph()
{
    graph.clear();
}

void SimpleGraph::addConnection(unsigned source, unsigned destination)
{
    std::pair<unsigned, unsigned> m_pair = std::make_pair(source, destination);
    std::vector<std::pair<unsigned, unsigned> >::iterator iter =
            std::find(graph.begin(), graph.end(), m_pair);
    if (iter == graph.end()) {
        graph.push_back(m_pair);
    }
}

bool SimpleGraph::removeConnection(unsigned source, unsigned destination)
{
    std::vector<std::pair<unsigned, unsigned> >::iterator iter =
            std::find(graph.begin(), graph.end(), std::make_pair(source, destination));
    if (iter != graph.end()) {
        graph.erase(iter);
        return true;
    }
    return false;
}

bool SimpleGraph::checkConnection(unsigned source, unsigned destination)
{
    std::vector<std::pair<unsigned, unsigned> >::iterator iter =
            std::find(graph.begin(), graph.end(), std::make_pair(source, destination));
    if (iter != graph.end()) {
        return true;
    }
    return false;
}

std::vector<std::pair<unsigned, unsigned> >::iterator SimpleGraph::getIterator()
{
    return graph.begin();
}

std::vector<std::pair<unsigned, unsigned> >::iterator SimpleGraph::getEnd()
{
    return graph.end();
}

void SimpleGraph::save(FILE* stream)
{
    unsigned size = graph.size();
    fwrite(&size, sizeof(unsigned), 1, stream);

    for (unsigned i = 0; i < size; ++i) {
        fwrite(&graph[i].first, sizeof(unsigned), 1, stream);
        fwrite(&graph[i].second, sizeof(unsigned), 1, stream);
    }
}

void SimpleGraph::load(FILE* stream)
{
    unsigned size, source, destination;
    fread(&size, sizeof(unsigned), 1, stream);

    for (unsigned i = 0; i < size; ++i) {
        fread(&source, sizeof(unsigned), 1, stream);
        fread(&destination, sizeof(unsigned), 1, stream);
        graph.push_back(std::make_pair(source, destination));
    }
}
