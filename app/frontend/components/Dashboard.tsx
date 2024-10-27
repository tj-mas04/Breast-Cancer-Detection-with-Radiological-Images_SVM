"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, BarChart2, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("upload")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [previousResults] = useState([
    { id: 1, date: "2023-05-01", result: "Benign" },
    { id: 2, date: "2023-05-15", result: "Malignant" },
    { id: 3, date: "2023-06-02", result: "Benign" },
  ])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePrediction = () => {
    // Simulating prediction
    setTimeout(() => {
      setPrediction(Math.random() > 0.5 ? "Benign" : "Malignant")
    }, 2000)
  }

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="w-64 bg-gray-800 p-4"
          >
            <div className="flex justify-between items-center mb-8">
              <h1 className="text-2xl font-bold">BC Dashboard</h1>
              <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(false)}>
                <X className="h-6 w-6" />
              </Button>
            </div>
            <nav>
              <Button
                variant={activeTab === "upload" ? "secondary" : "ghost"}
                className="w-full justify-start mb-2"
                onClick={() => setActiveTab("upload")}
              >
                <Upload className="mr-2 h-4 w-4" /> Upload & Predict
              </Button>
              <Button
                variant={activeTab === "results" ? "secondary" : "ghost"}
                className="w-full justify-start"
                onClick={() => setActiveTab("results")}
              >
                <BarChart2 className="mr-2 h-4 w-4" /> Previous Results
              </Button>
            </nav>
          </motion.aside>
        )}
      </AnimatePresence>

      <main className="flex-1 p-8">
        <Button variant="outline" size="icon" className="mb-4" onClick={() => setSidebarOpen(!sidebarOpen)}>
          <Menu className="h-6 w-6" />
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
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle>Upload Breast Cancer Image</CardTitle>
                  <CardDescription>Upload an image for prediction</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col items-center">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="image-upload"
                    />
                    <label
                      htmlFor="image-upload"
                      className="cursor-pointer bg-gray-700 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded"
                    >
                      Choose Image
                    </label>
                    {uploadedImage && (
                      <img src={uploadedImage} alt="Uploaded" className="mt-4 max-w-full h-auto max-h-64" />
                    )}
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="secondary" onClick={handlePrediction} disabled={!uploadedImage}>
                    Predict
                  </Button>
                  {prediction && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className={`font-bold ${
                        prediction === "Benign" ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      Prediction: {prediction}
                    </motion.div>
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
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle>Previous Results</CardTitle>
                  <CardDescription>View your past predictions</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    {previousResults.map((result, index) => (
                      <motion.div
                        key={result.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div className="flex justify-between items-center py-2">
                          <span>{result.date}</span>
                          <span
                            className={`font-bold ${
                              result.result === "Benign" ? "text-green-400" : "text-red-400"
                            }`}
                          >
                            {result.result}
                          </span>
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
      </main>
    </div>
  )
}