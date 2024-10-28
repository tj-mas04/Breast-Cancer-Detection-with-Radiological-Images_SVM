import LoginPage from '@/components/LoginPage'
import React from 'react'
import { Toaster } from 'sonner'

type Props = {}

const page = (props: Props) => {
  return (
    <div>
        <Toaster richColors position="top-center"/>
        <LoginPage/>
    </div>
  )
}

export default page