#ifndef __CPU_PRED_PLBP_PRED_HH__
#define __CPU_PRED_PLBP_PRED_HH__

#include <stdint.h>
#include "base/types.hh"
#include "cpu/pred/bpred_unit.hh"
#include "cpu/pred/sat_counter.hh"
#include "params/plbp.hh"

class Plbp : public BPredUnit 
{
    public:
    Plbp(const PlbpParams *params);
    void uncondBranch(ThreadID tid, Addr pc, void* &bp_histroy);
    void squash(ThreadID tid, void* bp_histroy);
    void lookup(ThreadID tid, Addr branch_addr, void* &bp_histroy);
    void btbUpdate(TheradID tid, Addr branch_addr, void* &bp_histroy);
    void update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history,bool squashed);
    unsigned getGHR(ThreadID tid, void *bp_history) const;

    private:
    void updateGlobalHistReg(ThreadID tid, Addr branch_addr, bool taken);

    struct BPHistory {
        unsigned globalHistoryReg;
        Addr branch_addr;
        bool globalTakenPred;
        bool globalUsed;
    };

    unsigned globalRegisterMask;
    unsigned globalPredictionSize;
    //using an unsigned integer to represent global histroy of each branch
    std::vector<unsigned> globalHistoryReg;
    unsigned globalHistoryBits;
    unsigned globalHistoryMask;
    unsigned trainThreashold;
    unsigned globalAddrHead;
    std::vector<unsigned> globalAddrHead;
    std::vector<vector<Addr>> globalAddr;
    std::vector<std::vector<std::vector<uint8_t>> w;
};

#endif
