"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { HeartPulse, ShieldCheck, Microscope, Users, ChevronRight, Github } from "lucide-react"
import Image from "next/image"
import Link from "next/link"

export default function Home() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-100">
      <header className="px-4 lg:px-6 h-14 flex items-center border-b border-gray-800">
        <Link className="flex items-center justify-center" href="#">
          <HeartPulse className="h-6 w-6 text-pink-500" />
          <span className="ml-2 text-lg font-bold">Breast Cancer Detection System</span>
        </Link>
        <nav className="ml-auto flex gap-4 sm:gap-6">
          <Link className="text-sm font-medium hover:text-pink-400 transition-colors" href="/login">
            Login
          </Link>
          <Link className="text-sm font-medium hover:text-pink-400 transition-colors" href="/signup">
            Sign Up
          </Link>
        </nav>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
          <motion.div className="container px-4 md:px-6" {...fadeIn}>
            <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                    Early Detection Saves Lives
                  </h1>
                  <p className="max-w-[600px] text-gray-400 md:text-xl">
                    Our advanced breast cancer detection system uses AI to provide accurate and early diagnosis,
                    increasing survival rates and improving patient outcomes.
                  </p>
                </div>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Button className="bg-pink-600 hover:bg-pink-700 text-white">Get Started</Button>
                  <Button variant="outline" className="text-pink-400 border-pink-400 hover:bg-pink-400 hover:text-gray-900">Learn More</Button>
                </div>
              </div>
              <div className="flex items-center justify-center">
                <Image
                  alt="Breast Cancer Detection System"
                  className="aspect-video overflow-hidden rounded-xl object-cover object-center"
                  height="310"
                  src="/placeholder.svg"
                  width="550"
                />
              </div>
            </div>
          </motion.div>
        </section>
        <motion.section 
          className="w-full py-12 md:py-24 lg:py-32 bg-gray-800"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="container mx-auto px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">Key Features</h2>
            <div className="grid gap-6 lg:grid-cols-3 lg:gap-12">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <ShieldCheck className="w-12 h-12 text-pink-500 mb-4" />
                  <CardTitle className="text-pink-400">Precise Accuracy</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-400">
                  Our AI-powered system provides industry-leading accuracy in breast cancer detection, minimizing false
                  positives and negatives.
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <Microscope className="w-12 h-12 text-pink-500 mb-4" />
                  <CardTitle className="text-pink-400">Advanced Imaging</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-400">
                  Utilizing cutting-edge imaging technology for detailed and comprehensive breast examinations.
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <Users className="w-12 h-12 text-pink-500 mb-4" />
                  <CardTitle className="text-pink-400">Patient-Centric</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-400">
                  Designed with patient comfort and privacy in mind, ensuring a positive experience throughout the
                  detection process.
                </CardContent>
              </Card>
            </div>
          </div>
        </motion.section>
        <motion.section 
          className="w-full py-12 md:py-24 lg:py-32"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="container mx-auto px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">About the Project</h2>
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12">
              <div>
                <h3 className="text-2xl font-bold mb-4 text-pink-400">Our Mission</h3>
                <p className="text-gray-400">
                  The Breast Cancer Detection System project aims to revolutionize early cancer detection using
                  cutting-edge AI technology. Our mission is to improve survival rates and patient outcomes by providing
                  accurate, accessible, and timely diagnoses.
                </p>
              </div>
              <div>
                <h3 className="text-2xl font-bold mb-4 text-pink-400">The Team</h3>
                <p className="text-gray-400">
                  Our diverse team of AI specialists, oncologists, and software engineers work tirelessly to develop
                  and improve our detection system. With decades of combined experience, we're committed to pushing the
                  boundaries of what's possible in medical technology.
                </p>
              </div>
            </div>
            <div className="mt-12 text-center">
              <Link href="https://github.com" className="inline-flex items-center text-pink-400 hover:text-pink-300">
                <Github className="mr-2" />
                View Project on GitHub
              </Link>
            </div>
          </div>
        </motion.section>
        <motion.section 
          className="w-full py-12 md:py-24 lg:py-32 bg-pink-600"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <div className="container mx-auto px-4 md:px-6">
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-white">
                  Join the Fight Against Breast Cancer
                </h2>
                <p className="max-w-[900px] text-gray-100 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed mx-auto">
                  Early detection is key. Schedule a screening today and take control of your health.
                </p>
              </div>
              <Button className="bg-white text-pink-600 hover:bg-gray-100">
                Schedule Screening
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        </motion.section>
      </main>
      <footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t border-gray-800">
        <p className="text-xs text-gray-500">
          © 2024 Breast Cancer Detection System. All rights reserved.
        </p>
        <nav className="sm:ml-auto flex gap-4 sm:gap-6">
          <Link className="text-xs hover:underline underline-offset-4 text-gray-500 hover:text-gray-400" href="#">
            Terms of Service
          </Link>
          <Link className="text-xs hover:underline underline-offset-4 text-gray-500 hover:text-gray-400" href="#">
            Privacy
          </Link>
        </nav>
      </footer>
    </div>
  )
}