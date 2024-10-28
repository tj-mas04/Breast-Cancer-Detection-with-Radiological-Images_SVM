"use client"

import { useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, BarChart2, X, ArrowLeftToLine, ArrowRightFromLine, ImageIcon, Loader2, Trash2, LogOut, Settings, BookOpen, HelpCircle, Brain } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Toaster, toast } from 'sonner'
import { useRouter } from "next/navigation"
import RiskLevelIndicator from "./RiskLevelIndicator"
import TrendAnalysis from "./TrendAnalysis"

const API_URL = "http://localhost:8000"

interface Screening {
  id: number
  image_path: string
  result: string
  confidence: number
  created_at: string
}

export default function Dashboard() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState("upload")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<Screening | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [previousResults, setPreviousResults] = useState<Screening[]>([])
  const [username, setUsername] = useState<string>("")

  useEffect(() => {
    const storedUsername = localStorage.getItem("username")
    if (storedUsername) {
      setUsername(storedUsername)
      fetchPreviousResults(storedUsername)
    } else {
      router.push('/')
    }
  }, [])

  const fetchPreviousResults = async (username: string) => {
    try {
      const response = await fetch(`${API_URL}/screenings/?username=${username}`)
      if (response.ok) {
        const data : any = await response.json()
        const reversedData = data.reverse();
        setPreviousResults(reversedData)
      } else {
        toast.error('Error fetching previous results')
      }
    } catch (error) {
      toast.error('Network error while fetching results')
    }
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('image/')) {
        processFile(file)
      } else {
        toast.error('Please upload an image file')
      }
    }
  }, [])

  const processFile = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
      setUploadedFile(file)
      setPrediction(null) 
    }
    reader.readAsDataURL(file)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      processFile(file)
    }
  }

  const handleDeleteImage = () => {
    setUploadedImage(null)
    setUploadedFile(null)
    setPrediction(null)
    const input = document.getElementById('image-upload') as HTMLInputElement
    if (input) {
      input.value = ''
    }
    toast.success('Image deleted successfully')
  }

  const handlePrediction = async () => {
    if (!uploadedFile || !username) {
      toast.error('Please upload an image and ensure you are logged in')
      return
    }

    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const response = await fetch(`${API_URL}/screenings?username=${username}`, {
        method: 'POST',
        body: formData,
        headers: {
          'username': username
        }
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const result = await response.json()
      setPrediction(result)
      toast.success('Prediction completed successfully')
      
      fetchPreviousResults(username)
    } catch (error) {
      toast.error('Error making prediction')
      console.error('Prediction error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const logoutUser = () => {
    localStorage.removeItem('username');
    localStorage.removeItem('userId');
    setActiveTab("logout")
    router.push('/login')
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="flex h-[100vh] bg-slate-950 text-slate-100 overflow-scroll">
      <Toaster position="top-right"/>

      <AnimatePresence>
        {sidebarOpen && (
           <motion.aside
           initial={{ x: -300 }}
           animate={{ x: 0 }}
           exit={{ x: -300 }}
           transition={{ type: "spring", stiffness: 300, damping: 30 }}
           className="fixed left-0 top-0 bottom-0 w-64 bg-slate-900/95 backdrop-blur-sm border-r border-slate-800 shadow-xl z-50"
         >
           <div className="flex flex-col h-full">
             <div className="p-4">
               <div className="flex justify-between items-center mb-8">
                 <h1 className="text-2xl font-bold text-slate-100">MediScan AI</h1>
                 <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(false)}>
                   <X className="h-6 w-6" />
                 </Button>
               </div>
               
               {username && (
                 <div className="mb-4 p-3 rounded-lg bg-slate-800/50 border border-slate-700">
                   <p className="capitalize flex items-center justify-center text-slate-200 font-bold text-lg ">
                     Welcome {username} !
                   </p>
                   {/* {userStats && (
                     <p className="text-sm text-slate-400 mt-1">
                       Total Scans: {userStats.total_screenings}
                     </p>
                   )} */}
                 </div>
               )}
             </div>
     
             <ScrollArea className="flex-grow px-4">
               <nav className="space-y-2">
                 <Button
                   variant={activeTab === "upload" ? "secondary" : "ghost"}
                   className="w-full justify-start"
                   onClick={() => setActiveTab("upload")}
                 >
                   <Upload className="mr-2 h-4 w-4" /> Upload & Predict
                 </Button>
     
                 <Button
                   variant={activeTab === "results" ? "secondary" : "ghost"}
                   className="w-full justify-start"
                   onClick={() => setActiveTab("results")}
                 >
                   <BarChart2 className="mr-2 h-4 w-4" /> Analysis History
                 </Button>
    
     
                 <Separator className="my-4" />
    

               </nav>
             </ScrollArea>
     
             <div className="p-4 border-t border-slate-800">
               <Button
                 variant="ghost"
                 className="w-full justify-start text-red-400 hover:text-red-300 hover:bg-red-500/10"
                 onClick={logoutUser}
               >
                 <LogOut className="mr-2 h-4 w-4" /> Logout
               </Button>
             </div>
           </div>
         </motion.aside>
        )}
      </AnimatePresence>

      <motion.main transition={{ type: "spring", stiffness: 300, damping: 30 }} className={` ${sidebarOpen ? 'ml-64' : 'ml-0'} flex-1 p-8 mb-10 `}>
        <Button 
          variant="outline" 
          size="icon" 
          className="mb-4 bg-slate-800 hover:bg-slate-900" 
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? 
            <ArrowLeftToLine className="h-6 w-6 text-slate-200 " /> : 
            <ArrowRightFromLine className="h-6 w-6 text-slate-200" />
          }
        </Button>

        <AnimatePresence mode="wait">
          {activeTab === "upload" && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="bg-slate-950 rounded-t-lg border-b border-slate-800">
                  <CardTitle className="text-3xl text-slate-100 font-bold text-center">
                    Breast Cancer Image Analysis
                  </CardTitle>
                  <CardDescription className="text-center text-lg text-slate-300">
                    Upload an image for instant AI-powered diagnosis
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center">
                    <div
                      onDragEnter={handleDragIn}
                      onDragLeave={handleDragOut}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                      className={`w-full max-w-xl h-64 relative rounded-lg border-2 border-dashed
                        ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700'}
                        ${uploadedImage ? 'p-2' : 'p-8'}
                        transition-colors duration-200 ease-in-out
                        flex items-center justify-center bg-slate-800/50`}
                    >
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="image-upload"
                      />
                      {uploadedImage ? (
                        <div className="relative w-full h-full">
                          <img 
                            src={uploadedImage} 
                            alt="Uploaded" 
                            className="w-full h-full object-contain rounded" 
                          />
                          <Button
                            variant="destructive"
                            size="icon"
                            className="absolute top-2 right-2"
                            onClick={handleDeleteImage}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center text-slate-400">
                          <ImageIcon className="w-16 h-16 mb-4" />
                          <p className="text-lg mb-2">Drag & drop your image here</p>
                          <p className="text-sm mb-4">or</p>
                          <label
                            htmlFor="image-upload"
                            className="cursor-pointer bg-slate-700 hover:bg-slate-600 text-slate-100 font-bold py-2 px-4 rounded transition-colors duration-200"
                          >
                            Browse Files
                          </label>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex flex-col space-y-4">
                  <div className="w-full flex justify-between items-center">
                    <div className="flex gap-2">
                      <Button 
                        variant="secondary" 
                        onClick={handlePrediction} 
                        disabled={!uploadedImage || isLoading}
                        className="w-32"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Processing
                          </>
                        ) : (
                          <>Predict</>
                        )}
                      </Button>
                      {uploadedImage && (
                        <label
                          htmlFor="image-upload"
                          className="cursor-pointer inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring bg-slate-700 text-slate-100 shadow hover:bg-slate-600/90 h-9 px-4"
                        >
                          Select Another Image
                        </label>
                      )}
                    </div>
                  </div>
                  
                  {prediction && (
                    <>
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="w-full p-4 rounded-lg bg-slate-800 border border-slate-700"
                    >
                      <h3 className="text-xl font-bold mb-2 text-slate-100">Latest Prediction Results:</h3>
                      <div className="space-y-2">
                        <p className={`text-lg font-semibold ${
                          prediction.result === "Benign" ? "text-green-400" : "text-red-400"
                        }`}>
                          Result: {prediction.result}
                        </p>
                        <p className="text-slate-300">
                          Confidence: {(prediction.confidence * 100).toFixed(2)}%
                        </p>
                        <p className="text-slate-300">
                          Date: {formatDate(prediction.created_at)}
                        </p>
                      </div>
                    </motion.div>
                    <RiskLevelIndicator
                    confidence={prediction.confidence} 
                    result={prediction.result} 
                  />
                  </>
                  )}
                </CardFooter>
              </Card>
            </motion.div>
          )}

          {activeTab === "results" && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="bg-slate-900 border-slate-800">
                <CardHeader className="bg-slate-950 rounded-t-lg border-b border-slate-800">
                  <CardTitle className="text-3xl font-bold text-center text-slate-100">
                    Previous Results
                  </CardTitle>
                  <CardDescription className="text-center text-lg text-slate-300">
                    View your analysis history
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="min-w-max min-h-max mt-4">
                  <TrendAnalysis  screenings={previousResults} />
                    {previousResults.map((result, index) => (
                      <motion.div
                        key={result.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div className="flex flex-col space-y-2 py-3 px-4 hover:bg-slate-800/50 rounded-lg">
                          <div className="flex justify-between items-center">
                            <span className="text-lg text-slate-200">{formatDate(result.created_at)}</span>
                            <span
                              className={`font-bold text-lg ${
                                result.result === "Benign" ? "text-green-400" : "text-red-400"
                              }`}
                            >
                              {result.result}
                            </span>
                          </div>
                          <div className="text-sm text-slate-400">
                            Confidence: {(result.confidence * 100).toFixed(2)}%
                          </div>
                        </div>
                        {index < previousResults.length - 1 && <Separator className="my-2" />}
                      </motion.div>
                    ))}


                  </ScrollArea>

                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.main>
    </div>
  )
}