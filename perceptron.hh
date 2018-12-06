#ifndef __CPU_PRED_PERCEPTRON_PRED_HH__
#define __CPU_PRED_PERCEPTRON_PRED_HH__

#include "base/types.hh"
#include "cpu/pred/bpred_unit.hh"
#include "cpu/pred/sat_counter.hh"
#include "params/Perceptron.hh"

class Perceptron : public BPredUnit 
{
    public:
    Perceptron(const PerceptronParams *params);
    void uncondBranch(ThreadID tid, Addr pc, void* &bp_histroy);
    void squash(ThreadID tid, void* bp_histroy);
    void lookup(ThreadID tid, Addr branch_addr, void* &bp_histroy);
    void btbUpdate(TheradID tid, Addr branch_addr, void* &bp_histroy);
    void update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history,bool squashed);
    unsigned getGHR(ThreadID tid, void *bp_history) const;

    private:
    void updateGlobalHistReg(ThreadID tid, bool taken);

    struct BPHistory {
        unsigned globalHistoryReg;
        bool globalTakenPred;
        bool globalUsed;
    }

    unsigned globalRegisterMask;
    unsigned globalPredictionSize;
    //using an unsigned integer to represent global histroy of each branch
    std::vector<unsigned> globalHistoryReg;
    unsigned globalHistoryBits;
    unsigned globalHistoryMask;
    unsigned numOfPerceptrons;
    unsigned trainThreashold;
    //weight vectors
    std::vector<std::vector<unsigned>> w;
}