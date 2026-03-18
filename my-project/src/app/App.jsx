import { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import { TuteeCard } from '../components/TuteeCard';
import svgPaths from '../imports/svg-gdlx7jt9bs';
import { imgUnion } from '../imports/svg-864lg';


export default function App() {

  const [tutees, setTutees] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedSummary, setSelectedSummary] = useState(null); // Stores the AI text
  const [isAiLoading, setIsAiLoading] = useState(false);

  // Replace '2' with the actual logged-in tutor's ID
  const currentTutorId = 1; 

  useEffect(() => {
    fetch(`http://localhost:5000/api/tutees/${currentTutorId}`)
      .then(res => res.json())
      .then(data => {
        // Map the DB fields to match your TuteeCard expectations if necessary
        const formattedData = data.map(t => ({
          ...t,
          name: `${t.first_name} ${t.last_name}`,
          initial: t.first_name[0],
          proficiencyLevel: t.proficiency_level === 1 ? 'Beginner' : 'Intermediate' 
        }));
        setTutees(formattedData);
        setLoading(false);
      })
      .catch(err => console.error("Error fetching tutees:", err));
  }, []);

  const filteredTutees = tutees.filter(tutee =>
    tutee.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (loading) {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <p className="text-[#36c231] font-semibold animate-pulse">Loading Tutees...</p>
    </div>
  );
}

const handleGenerateSummary = async (tutee) => {
    setIsAiLoading(true);
    
    // 1. Build the specific prompt using DB data
    const prompt = `
        You are an ai assistant to help tutors teaching Filipino Sign Language get a summary of their 
        tutee's capabilities and current progress with their learning materials as well as their previous 
        attempts at  quizzes. You will be provided the tutee to analyze, alongside all the information from 
        their completed lessons and quizzes. For this particular tutee this is the information provided:

        TUTEE PROFILE:
          - Name: ${tutee.name}
          - Level: ${tutee.proficiencyLevel}
                
        CURRICULUM CONTEXT:
          - Active Module: "${tutee.currentModule}"
          Description: ${tutee.currentModuleDesc || 'No description available'}
          - Active Lesson: "${tutee.currentLesson}"
          Description: ${tutee.currentLessonDesc || 'No description available'}

       PERFORMANCE DATA:
          - Quiz Success: ${tutee.completedQuizzes} passed out of ${tutee.totalQuizAttempts} attempts.

       TASK:
          Provide a summary for the tutor. The "Areas to work on" and "Suggested assignments" must be 
          specifically tailored to the content described in the Active Module and Lesson descriptions.
          Remember to use the context of Filipino Sign Language when you provide your overview, areas
          to work on, and suggested assignments.

        FORMAT:
         📝Overview: (1-2 sentences MAX)
         ❗Areas to work on: (Max 3 bullets)
         📚Suggested assignments: (Max 3 bullets)
    `;

    try {
      // 2. Call Puter.js (window.puter ensures React sees the global script)
      const response = await window.puter.ai.chat(prompt, { model: "gpt-5-nano" });
      
      // 3. Save the result to show in an overlay
      setSelectedSummary({
        name: tutee.name,
        content: response.toString() 
      });
    } catch (error) {
      console.error("AI Error:", error);
      alert("Could not reach the AI assistant.");
    } finally {
      setIsAiLoading(false);
    }
  };

  
  return (
    <div className="bg-white relative w-full min-h-screen">
      {/* Status Bar */}
      <div className="bg-[#36c231] h-[54px] w-full px-6 flex items-center justify-between">
        <div className="font-bold text-[17px] text-[#1c1b1f]">9:41</div>
        <div className="flex items-center gap-2">
          {/* Signal indicators */}
          <div className="h-[12px] w-[19px]">
            <svg className="w-full h-full" fill="none" viewBox="0 0 19.1955 12.2373">
              <path clipRule="evenodd" d={svgPaths.p4a80080} fill="#1C1B1F" fillRule="evenodd" />
            </svg>
          </div>
          <div className="h-[12px] w-[17px]">
            <svg className="w-full h-full" fill="none" viewBox="0 0 17.1357 12.3412">
              <path clipRule="evenodd" d={svgPaths.p268bd180} fill="#1C1B1F" fillRule="evenodd" />
            </svg>
          </div>
          <div className="border-[#1c1b1f] border-[1.1px] border-solid h-[13px] w-[25px] opacity-35 rounded-[4.3px]" />
          <div className="bg-[#1c1b1f] h-[9px] w-[21px] rounded-[2.5px]" />
        </div>
      </div>

      {/* Top Navigation Bar */}
      <div className="bg-[#36c231] h-[68px] w-full px-6 flex items-center justify-between">
        <p className="font-semibold text-[20px] text-white">Tutor Dashboard</p>
        <div className="flex items-center gap-3">
          {/* AI Assistant Ready Indicator */}
          <div className="flex items-center gap-2 bg-white/20 rounded-full px-3 py-1.5">
            <div className="size-2 rounded-full bg-white animate-pulse" />
            <span className="text-white text-[12px] font-medium">Assistant Ready</span>
          </div>
          <button className="size-[44px] flex items-center justify-center">
            <Search className="size-5 text-white" />
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="w-full px-4 pt-4 pb-2">
        <div className="flex items-center justify-between gap-4">
          {/* Appointments */}
          <div className="flex flex-col items-center gap-2">
            <p className="text-[12px] text-[#666]">Appointments</p>
            <div className="size-[48px]">
              <svg className="w-full h-full" fill="none" viewBox="0 0 48 48">
                <path d="M16 4V12" stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d="M32 4V12" stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d={svgPaths.p3a671570} stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d="M6 20H42" stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
              </svg>
            </div>
            <p className="text-[20px] text-[#666] font-bold">8</p>
          </div>

          {/* Tutees */}
          <div className="flex flex-col items-center gap-2">
            <p className="text-[12px] text-[#666]">Tutees</p>
            <div className="size-[44px]">
              <svg className="w-full h-full" fill="none" viewBox="0 0 44 44">
                <path d={svgPaths.p38a7380} stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d={svgPaths.p1831bb80} stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d={svgPaths.p2ba2e600} stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d={svgPaths.p303d0d80} stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
              </svg>
            </div>
            <p className="text-[20px] text-[#666] font-bold">
              {tutees.length} {/* This now shows the real count from the DB */}
            </p>
          </div>

          {/* Assignments */}
          <div className="flex flex-col items-center gap-2">
            <p className="text-[12px] text-[#666]">Assignments</p>
            <div className="size-[42px]">
              <svg className="w-full h-full" fill="none" viewBox="0 0 42 42">
                <path d="M21 12.25V36.75" stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
                <path d={svgPaths.pa736380} stroke="#36C231" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.99917" />
              </svg>
            </div>
            <p className="text-[20px] text-[#666] font-bold">1</p>
          </div>
        </div>
      </div>

      {/* Tab Buttons */}
      <div className="w-full px-4 py-3 overflow-x-auto">
        <div className="flex gap-2 min-w-max">
          <button className="bg-[#e3e2d9] h-[35px] rounded-[15px] px-4 flex items-center gap-2 whitespace-nowrap">
            <div className="size-4">
              <svg className="w-full h-full" fill="none" viewBox="0 0 16 16">
                <path d="M5.33111 1.33278V3.99833" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
                <path d="M10.6622 1.33278V3.99833" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
                <path d={svgPaths.p253a8b00} stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
                <path d="M1.99917 6.66389H13.9942" stroke="black" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
              </svg>
            </div>
            <span className="text-[13px] font-semibold text-black">Appointments</span>
          </button>

          <button className="bg-[#36c231] h-[35px] rounded-[15px] px-4 flex items-center gap-2 whitespace-nowrap">
            <div className="size-4">
              <svg className="w-full h-full" fill="none" viewBox="0 0 16 16">
                <path d={svgPaths.p34460200} stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
                <path d={svgPaths.p1aacaf00} stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
                <path d={svgPaths.p3d3b16a0} stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
                <path d={svgPaths.p3b513d00} stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.33278" />
              </svg>
            </div>
            <span className="text-[13px] font-semibold text-white">My Tutees</span>
          </button>

          <button className="bg-[#e3e2d9] h-[35px] rounded-[15px] px-4 whitespace-nowrap">
            <span className="text-[13px] font-semibold text-black">Assignments</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="w-full px-4 pb-24">
        {/* Pending Applications */}
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-[16px] text-black">Pending Applications</h2>
          <span className="text-[13px] text-[#ffa726]">1 new</span>
        </div>

        {/* Search Tutees */}
        <div className="mb-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search Tutees"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full h-[40px] bg-white border border-[#e3e2d9] rounded-[12px] pl-4 pr-10 text-[14px] text-black placeholder:text-[#666] focus:outline-none focus:border-[#36c231]"
            />
            <Search className="absolute right-3 top-1/2 -translate-y-1/2 size-4 text-[#666]" />
          </div>
        </div>

{/* Example: Where you map your tutees, pass the function */}
      <div className="space-y-3">
        {tutees.map((tutee) => (
          <div key={tutee.id} className="relative">
             <TuteeCard tutee={tutee} />
             {/* Add a button or make the card clickable to trigger handleGenerateSummary(tutee) */}
             <button 
                onClick={() => handleGenerateSummary(tutee)}
                className="absolute top-4 right-4 bg-green-100 text-green-700 px-3 py-1 rounded-full text-xs font-bold"
             >
               {isAiLoading ? 'Analyzing...' : 'AI Summary'}
             </button>
          </div>
        ))}
      </div>

      {/* 3. The Figma-style Overlay (Modal) */}
      {selectedSummary && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-end sm:items-center justify-center p-4">
          <div className="bg-white w-full max-w-md rounded-t-[30px] sm:rounded-[30px] p-6 shadow-2xl animate-in slide-in-from-bottom duration-300">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-800">AI Summary - {selectedSummary.name}</h2>
              <button onClick={() => setSelectedSummary(null)} className="text-gray-400 text-2xl">&times;</button>
            </div>
            
            <div className="prose prose-sm max-h-[60vh] overflow-y-auto">
              {/* This renders the AI text. You can parse the markdown if needed */}
              <p className="whitespace-pre-wrap text-gray-600 leading-relaxed">
                {selectedSummary.content}
              </p>
            </div>

            <button 
              onClick={() => setSelectedSummary(null)}
              className="w-full mt-6 bg-[#36c231] text-white py-3 rounded-xl font-bold"
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
    
    {/* Bottom Navigation */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-[#e3e2d9] h-[56px]">
        <div className="flex items-center justify-around h-full px-4">
          <button className="flex flex-col items-center gap-1">
            <div className="size-6">
              <svg className="w-full h-full" fill="none" viewBox="0 0 20 18">
                <path d={svgPaths.p133bd200} fill="#BDBDBD" stroke="#BDBDBD" strokeWidth="1.5" />
              </svg>
            </div>
          </button>

          <button className="flex flex-col items-center gap-1">
            <div className="size-6">
              <svg className="w-full h-full" fill="none" viewBox="0 0 24 20">
                <path d={svgPaths.pd3b8c80} fill="#BDBDBD" />
              </svg>
            </div>
          </button>

          <button className="flex flex-col items-center gap-1">
            <div className="size-6">
              <svg className="w-full h-full" fill="none" viewBox="0 0 21 21">
                <path clipRule="evenodd" d={svgPaths.p8e5d400} fill="#36C231" fillRule="evenodd" />
                <path clipRule="evenodd" d={svgPaths.pd89b300} fill="#36C231" fillRule="evenodd" />
              </svg>
            </div>
          </button>

          <button className="flex flex-col items-center gap-1">
            <div className="size-6">
              <svg className="w-full h-full" fill="none" viewBox="0 0 16 8">
                <path d={svgPaths.p8958a00} fill="#A09CAB" />
                <path d={svgPaths.p346da800} fill="#A09CAB" />
              </svg>
            </div>
          </button>
        </div>
      </div>

      {/* Home Indicator */}
      <div className="fixed bottom-0 left-0 right-0 bg-white h-[21px] flex items-center justify-center">
        <div className="bg-[#1c1b1f] h-[5px] rounded-[100px] w-[139px]" />
      </div>
    </div>


  );

  
}
