import { useState } from 'react';
import { Sparkles, X } from 'lucide-react';

interface Tutee {
  id: string;
  name: string;
  initial: string;
  proficiencyLevel: string;
  currentModule: string;
  currentLesson: string;
  completedQuizzes: number;
  totalQuizAttempts: number;
  aiSummary: {
    overview: string;
    areasToImprove: string[];
    suggestedAssignments: string[];
  };
}

interface TuteeCardProps {
  tutee: Tutee;
}

export function TuteeCard({ tutee }: TuteeCardProps) {
  const [showAISummary, setShowAISummary] = useState(false);

  return (
    <>
      <div className="bg-[#e3e2d9] relative rounded-[20px] w-full p-4">
        <div className="flex items-start gap-3 w-full">
          {/* Avatar */}
          <div className="bg-[#36c231] rounded-full size-[50px] flex items-center justify-center shrink-0">
            <p className="font-semibold text-[20px] text-white">{tutee.initial}</p>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Name and AI Icon */}
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="font-semibold text-[15px] text-black">{tutee.name}</p>
                <p className="font-normal text-[12px] text-[#666] mt-0.5">{tutee.proficiencyLevel}</p>
              </div>
              
              {/* AI Summary Button */}
              <button
                onClick={() => setShowAISummary(true)}
                className="flex items-center justify-center size-8 rounded-full bg-[#36c231] hover:bg-[#2da528] transition-colors"
                aria-label="View AI Summary"
              >
                <Sparkles className="size-4 text-white" />
              </button>
            </div>

            {/* Info Grid */}
            <div className="grid grid-cols-2 gap-2 text-[12px]">
              <div>
                <p className="text-[#666]">Current Module</p>
                <p className="font-semibold text-black">{tutee.currentModule}</p>
              </div>
              <div>
                <p className="text-[#666]">Current Lesson</p>
                <p className="font-semibold text-black">{tutee.currentLesson}</p>
              </div>
              <div>
                <p className="text-[#666]">Completed Quizzes</p>
                <p className="font-semibold text-black">{tutee.completedQuizzes}</p>
              </div>
              <div>
                <p className="text-[#666]">Quiz Attempts</p>
                <p className="font-semibold text-black">{tutee.totalQuizAttempts}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AI Summary Modal */}
      {showAISummary && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-end justify-center">
          <div className="bg-white w-full max-w-[375px] rounded-t-[24px] p-6 animate-in slide-in-from-bottom duration-300">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sparkles className="size-5 text-[#36c231]" />
                <h2 className="font-semibold text-[18px] text-black">AI Summary - {tutee.name}</h2>
              </div>
              <button
                onClick={() => setShowAISummary(false)}
                className="size-8 flex items-center justify-center rounded-full hover:bg-[#e3e2d9] transition-colors"
              >
                <X className="size-5 text-[#666]" />
              </button>
            </div>

            {/* Content */}
            <div className="space-y-4 max-h-[60vh] overflow-y-auto">
              {/* Overview */}
              <div>
                <h3 className="font-semibold text-[14px] text-black mb-2">Overview</h3>
                <p className="text-[13px] text-[#666] leading-relaxed">{tutee.aiSummary.overview}</p>
              </div>

              {/* Areas to Improve */}
              <div>
                <h3 className="font-semibold text-[14px] text-black mb-2">Areas to Work On</h3>
                <ul className="space-y-2">
                  {tutee.aiSummary.areasToImprove.map((area, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <div className="size-1.5 rounded-full bg-[#ffa726] mt-1.5 shrink-0" />
                      <p className="text-[13px] text-[#666] flex-1">{area}</p>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Suggested Assignments */}
              <div>
                <h3 className="font-semibold text-[14px] text-black mb-2">Suggested Assignments</h3>
                <div className="space-y-2">
                  {tutee.aiSummary.suggestedAssignments.map((assignment, index) => (
                    <div
                      key={index}
                      className="bg-[#e3e2d9] rounded-[12px] p-3"
                    >
                      <p className="text-[13px] text-black">{assignment}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Close Button */}
            <button
              onClick={() => setShowAISummary(false)}
              className="w-full mt-6 bg-[#36c231] text-white font-semibold text-[14px] py-3 rounded-[12px] hover:bg-[#2da528] transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}
