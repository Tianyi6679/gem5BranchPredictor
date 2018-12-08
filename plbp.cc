#include "cpu/pred/plbp.hh"

#include "base/bitfield.hh"
#include "base/intmath.hh"
#define MIN_WEIGHT -128
#define MAX_WEIGHT 127
#define N 256
#define M 256

Plbp::Plbp(const PlbpParams* params)
    : BPredUnit(params),
        globalHistoryBits(ceilLog2(params->globalPredictorSize)),
        globalPredictionSize(params->globalPredictorSize),
        globalHistoryReg(params->numThreads, 0),
        globalAddrHead(params->numThreads, 0),
        globalAddr(params->numThreads, std::vector<Addr>(params->globalPredictorSize,0)){
    if (!isPowerOf2(globalPredictionSize)){
        fatal("Invalid global predictor size! \n");
    }
    historyRegisterMask = mask(globalHistoryBits);
    trainThreashold = 1.93*globalPredictionSize + 14;
    w.assign(N, 
	     std::vector<std::vector<int8_t>>(M , 
                    std::vector<int8_t>(globalPredictionSize + 1, 0)));
}

void Plbp::uncondBranch(ThreadID tid, Addr pc, void* &bp_history){
    BPHistory* history = new BPHistory;
    unsigned head = globalAddrHead[tid];
    history->globalHistoryReg = globalHistoryReg[tid];
    history->globalTakenPred = true;
    history->globalUsed = true;
    history->branch_addr = globalAddr[tid][head];
    history->addrHead = head;
    bp_history = static_cast<void*>(history);
    updateGlobalHistReg(tid, pc, true);
}

void Plbp::squash(ThreadID tid, void *bp_history){
    BPHistory *history = static_cast<BPHistory*>(bp_history);
    globalHistoryReg[tid] = history->globalHistoryReg;
    globalAddrHead[tid] = history->addrHead;
    globalAddr[tid][globalAddrHead[tid]] = history->branch_addr;
    delete history;
}

bool Plbp::lookup(ThreadID tid, Addr branch_addr, void* &bp_history){
    int tableIndex = branch_addr % N;
    unsigned thread_history = globalHistoryReg[tid];
    unsigned head = globalAddrHead[tid];
    int bias = w[tableIndex][0][0];
    int dot_product = 0;
    for (int i = 1; i<=globalPredictionSize; i++){
        int globalAddrIndex = ((head - (i-1)) % globalPredictionSize + globalPredictionSize) % globalPredictionSize;
        int addr = globalAddr[tid][globalAddrIndex] % M;
        if ((thread_history >> (i-1)) & 1 ){
            dot_product += w[tableIndex][addr][i]; 
        }
        else{
            dot_product -= w[tableIndex][addr][i]; 
        }
    }
    // if y >=0, branch is predicted to be taken
    bool taken = (dot_product+bias) >= 0 ;
    
    BPHistory *history = new BPHistory;
    history->globalHistoryReg = globalHistoryReg[tid];
    history->globalTakenPred = taken;
    history->branch_addr = globalAddr[tid][head];
    history->addrHead = head;
    bp_history = static_cast<void*>(history);
    updateGlobalHistReg(tid, branch_addr, taken);
    return taken;
}

void Plbp::btbUpdate(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    globalHistoryReg[tid] &= (historyRegisterMask & ~ULL(1));
}

void Plbp::update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history, bool squashed){
    assert(bp_history);

    BPHistory *history = static_cast<BPHistory*>(bp_history);
    int tableIndex = branch_addr % N;
    if (squashed){
        globalHistoryReg[tid] = history->globalHistoryReg;
        globalAddrHead[tid] = history->addrHead;
        globalAddr[tid][globalAddrHead[tid]] = history->branch_addr;
    }

    unsigned thread_history = globalHistoryReg[tid];
    unsigned head = globalAddrHead[tid];
    int bias = w[tableIndex][0][0];
    int dot_product = 0;
    for (int i = 1; i<=globalPredictionSize; i++){
        int globalAddrIndex = ((head - (i-1)) % globalPredictionSize + globalPredictionSize) % globalPredictionSize;
        int addr = globalAddr[tid][globalAddrIndex] % M;
        if ((thread_history >> (i-1)) & 1 ){
            dot_product += w[tableIndex][addr][i]; 
        }
        else{
            dot_product -= w[tableIndex][addr][i]; 
        }
    }
    //training algorithm
    if (squashed || (abs(dot_product+bias) <= trainThreashold)){
        if (taken) w[tableIndex][0][0] += (w[tableIndex][0][0] < MAX_WEIGHT) ? 1:0;
        else w[tableIndex][0][0] -= (w[tableIndex][0][0] > MIN_WEIGHT) ? 1:0;
        for (int i = 1; i<=globalPredictionSize; i++){
            int globalAddrIndex = ((head - (i-1)) % globalPredictionSize + globalPredictionSize) % globalPredictionSize;
            int addr = globalAddr[tid][globalAddrIndex] % M;
            if (((thread_history >> (i-1)) & 1) == taken ){
                w[tableIndex][addr][i] += (w[tableIndex][addr][i] < MAX_WEIGHT) ? 1:0; 
            }
            else{
                w[tableIndex][addr][i] -= (w[tableIndex][addr][i] > MIN_WEIGHT) ? 1:0; 
            }
        }
    }
    updateGlobalHistReg(tid, branch_addr, taken);
}

unsigned Plbp::getGHR(ThreadID tid, void *bp_history) const
{
  return static_cast<BPHistory *>(bp_history)->globalHistoryReg;
}

void Plbp::updateGlobalHistReg(ThreadID tid, Addr branch_addr, bool taken){
    globalHistoryReg[tid] = taken ? (globalHistoryReg[tid] << 1) | 1 : (globalHistoryReg[tid] << 1);
    globalHistoryReg[tid] &= historyRegisterMask;
    globalAddr[tid][globalAddrHead[tid]] = branch_addr;
    globalAddrHead[tid] = (globalAddrHead[tid] + 1) % globalPredictionSize; 
}

Plbp* PlbpParams::create()
{
  return new Plbp(this);
}


