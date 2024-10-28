"use client"
import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { HeartPulse } from "lucide-react"
import axios from "axios"
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

export default function LoginPage() {
  const router = useRouter();
  const [username, setusername] = useState('')
  const [password, setPassword] = useState('')
  const [loading,setLoading] = useState<boolean>(false);



  /* 
  {
    "message": "Login successful",
    "username": "kishor",
    "userId": 2
  }
  */

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    try {
      setLoading(true);
      const res = await axios.post('http://localhost:8000/login',{
        username,
        password
      })
      
      if(res.status == 200 || res.status == 201) {
        localStorage.setItem('userId',res.data.userId);
        localStorage.setItem('username',res.data.username)
        localStorage.setItem('isLoggedIn','true');
        router.push("/dashboard")
      }
      else if(res.status == 401 || res.status == 404){
        console.log(res.data)
        toast.error(res.data.detail)
        return;
      }
    } catch (error : any) {
      if (error.response?.status === 401 || error.response?.status === 404) {
        toast.error(error.response?.data?.detail || "Invalid credentials or user not found");
      } else {
        toast.error("Error, Try Again please");
      }
      setLoading(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col justify-center items-center p-4">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <HeartPulse className="mx-auto h-12 w-12 text-pink-500" />
          <h2 className="mt-6 text-3xl font-extrabold text-white">MediScan AI</h2>
          <p className="mt-2 text-sm text-gray-400">Early detection saves lives. Log in to access your account.</p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <Label htmlFor="username-address" className="sr-only">Username</Label>
              <Input
                id="text-address"
                name="text"
                type="text"
                autoComplete="text"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-700 placeholder-gray-500 text-white rounded-t-md focus:outline-none focus:ring-pink-500 focus:border-pink-500 focus:z-10 sm:text-sm bg-gray-800"
                placeholder="Username"
                value={username}
                onChange={(e) => setusername(e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="password" className="sr-only">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-700 placeholder-gray-500 text-white rounded-b-md focus:outline-none focus:ring-pink-500 focus:border-pink-500 focus:z-10 sm:text-sm bg-gray-800"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                className="h-4 w-4 text-pink-600 focus:ring-pink-500 border-gray-700 rounded bg-gray-800"
              />
              <Label htmlFor="remember-me" className="ml-2 block text-sm text-gray-400">
                Remember me
              </Label>
            </div>

            <div className="text-sm">
              <a href="#" className="font-medium text-pink-500 hover:text-pink-400">
                Forgot your password?
              </a>
            </div>
          </div>

          <div>
            <Button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-pink-600 hover:bg-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500" disabled={loading}
            >
              {loading ? 'Loading..' : 'Log In'}
            </Button>
          </div>
        </form>
        <p className="mt-2 text-center text-sm text-gray-400">
          Don't have an account?{' '}
          <a href="/signup" className="font-medium text-pink-500 hover:text-pink-400">
            Sign up here
          </a>
        </p>
      </div>
    </div>
  )
}