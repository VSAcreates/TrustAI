"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { AlertCircle, Clock, Filter, RefreshCw, Search, Shield, User, CreditCard, Info } from "lucide-react"
import { toast } from "@/components/ui/use-toast"

// Backend API URL
const API_URL = "http://localhost:5000/api";

export default function FraudDashboard() {
  // State for dashboard
  const [riskThreshold, setRiskThreshold] = useState(70)
  const [timeRange, setTimeRange] = useState("24h")
  const [transactions, setTransactions] = useState([])
  const [alerts, setAlerts] = useState([])
  const [stats, setStats] = useState({
    transaction_count: { last_hour: 0, last_day: 0, total: 0 },
    fraud_count: { last_hour: 0, last_day: 0, total: 0 },
    risk_levels: { high: 0, medium: 0, low: 0 },
    alert_count: 0,
    current_risk_level: 30
  })
  const [loading, setLoading] = useState(true)
  const [accountInfo, setAccountInfo] = useState({
    name: "John Alen",
    accountNumber: "165489556489465",
    balance: 85000.00
  })

  // Fetch data from API
  const fetchData = async () => {
    try {
      // Convert timeRange to hours for API
      const hours = timeRangeToHours(timeRange)
      
      // Get transactions
      const txResponse = await fetch(`${API_URL}/transactions/recent?hours=${hours}`)
      const txData = await txResponse.json()
      setTransactions(txData)
      
      // Get alerts
      const alertResponse = await fetch(`${API_URL}/alerts`)
      const alertData = await alertResponse.json()
      setAlerts(alertData)
      
      // Get dashboard stats
      const statsResponse = await fetch(`${API_URL}/dashboard/stats`)
      const statsData = await statsResponse.json()
      setStats(statsData)
      
      setLoading(false)
    } catch (error) {
      console.error("Error fetching data:", error)
      toast({
        title: "Error",
        description: "Failed to fetch data from the server",
        variant: "destructive"
      })
    }
  }

  // Helper to convert timeRange to hours
  const timeRangeToHours = (range) => {
    switch(range) {
      case "1h": return 1
      case "6h": return 6
      case "24h": return 24
      case "7d": return 168
      default: return 24
    }
  }

  // Simulate new transaction
  const simulateTransaction = async (injectFraud = false) => {
    try {
      const response = await fetch(`${API_URL}/simulate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          inject_fraud: injectFraud
        })
      })
      
      const newTransaction = await response.json()
      
      // Add new transaction to state
      setTransactions(prev => [newTransaction, ...prev])
      
      // If it's fraudulent, add to alerts
      if (newTransaction.is_fraud) {
        // Refresh alerts
        const alertResponse = await fetch(`${API_URL}/alerts`)
        const alertData = await alertResponse.json()
        setAlerts(alertData)
        
        // Show toast notification
        toast({
          title: "Fraud Alert",
          description: `High-risk transaction detected: ${newTransaction.transaction_id}`,
          variant: "destructive"
        })
      }
      
      // Update stats
      fetchData()
      
    } catch (error) {
      console.error("Error simulating transaction:", error)
      toast({
        title: "Error",
        description: "Failed to simulate transaction",
        variant: "destructive"
      })
    }
  }

  // Initial data fetch
  useEffect(() => {
    fetchData()
    
    // Set up polling for new data every 10 seconds
    const interval = setInterval(() => {
      fetchData()
    }, 10000)
    
    return () => clearInterval(interval)
  }, [timeRange])

  // Helper function to format relative time
  const getRelativeTime = (timestamp) => {
    const now = Math.floor(Date.now() / 1000)
    const diff = now - timestamp
    
    if (diff < 60) return "Just now"
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`
    if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`
    return `${Math.floor(diff / 86400)} days ago`
  }
  
  // Get risk color based on score
  const getRiskColor = (score) => {
    if (score >= 70) return "bg-red-500"
    if (score >= 50) return "bg-yellow-500"
    return "bg-green-500"
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b bg-background px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-semibold">Fraud Detection Dashboard</h1>
          </div>
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm" onClick={() => simulateTransaction()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Random Transaction
            </Button>
            <Button variant="destructive" size="sm" onClick={() => simulateTransaction(true)}>
              <AlertCircle className="mr-2 h-4 w-4" />
              Simulate Fraud
            </Button>
            <Select defaultValue={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-[120px]">
                <Clock className="mr-2 h-4 w-4" />
                <SelectValue placeholder="Time Range" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">Last Hour</SelectItem>
                <SelectItem value="6h">Last 6 Hours</SelectItem>
                <SelectItem value="24h">Last 24 Hours</SelectItem>
                <SelectItem value="7d">Last 7 Days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </header>

      <main className="flex-1 p-4 md:p-6">
        <div className="grid gap-6 md:grid-cols-3">
          {/* Left Panel - Risk Assessment */}
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Risk Meter</CardTitle>
                <CardDescription>Current fraud risk level</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center space-y-4">
                  <div className="relative h-40 w-40">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <div className="text-3xl font-bold">{stats.current_risk_level}%</div>
                        <div className="text-sm text-muted-foreground">Risk Score</div>
                      </div>
                    </div>
                    <svg className="h-full w-full" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="45" fill="none" stroke="#e2e8f0" strokeWidth="10" />
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke={stats.current_risk_level >= 70 ? "#ef4444" : stats.current_risk_level >= 40 ? "#f59e0b" : "#10b981"}
                        strokeWidth="10"
                        strokeDasharray="282.7"
                        strokeDashoffset={282.7 - (282.7 * stats.current_risk_level / 100)}
                        transform="rotate(-90 50 50)"
                      />
                    </svg>
                  </div>
                  <div className="w-full space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Risk Threshold</span>
                      <span>{riskThreshold}%</span>
                    </div>
                    <Slider
                      value={[riskThreshold]}
                      min={0}
                      max={100}
                      step={1}
                      onValueChange={(value) => setRiskThreshold(value[0])}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Account Information Card */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Account Information</CardTitle>
                <CardDescription>Current account details</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-3 rounded-lg border p-3">
                    <User className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm text-muted-foreground">Account Holder</p>
                      <p className="font-medium">{accountInfo.name}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 rounded-lg border p-3">
                    <CreditCard className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm text-muted-foreground">Account Number</p>
                      <p className="font-medium">{accountInfo.accountNumber}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 rounded-lg border p-3">
                    <Info className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm text-muted-foreground">Current Balance</p>
                      <p className="font-medium">${accountInfo.balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Alerts Card */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Active Alerts</CardTitle>
                <CardDescription>Suspicious activity requiring attention</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {alerts.length === 0 ? (
                    <div className="rounded-lg border border-dashed p-8 text-center">
                      <Shield className="mx-auto h-10 w-10 text-muted-foreground" />
                      <h3 className="mt-2 font-medium">No Active Alerts</h3>
                      <p className="text-sm text-muted-foreground">All transactions are within normal parameters</p>
                    </div>
                  ) : (
                    alerts.slice(0, 3).map((alert) => (
                      <Alert key={alert.id} variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Fraud Alert #{alert.id}</AlertTitle>
                        <AlertDescription className="text-sm">
                          {alert.description}
                          <div className="mt-1 text-xs text-muted-foreground">
                            {getRelativeTime(alert.timestamp)}
                          </div>
                        </AlertDescription>
                      </Alert>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Middle Panel - Transaction List */}
          <div className="space-y-6">
            <Card className="col-span-2">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Recent Transactions</CardTitle>
                    <CardDescription>Latest financial activity</CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm">
                      <Filter className="mr-2 h-4 w-4" />
                      Filter
                    </Button>
                    <Button variant="outline" size="sm">
                      <Search className="mr-2 h-4 w-4" />
                      Search
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center py-10">
                    <div className="text-center">
                      <RefreshCw className="mx-auto h-8 w-8 animate-spin text-muted-foreground" />
                      <p className="mt-2 text-sm text-muted-foreground">Loading transactions...</p>
                    </div>
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Transaction ID</TableHead>
                        <TableHead>Time</TableHead>
                        <TableHead>Amount</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Risk Score</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {transactions.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={5} className="text-center py-8">
                            <p className="text-muted-foreground">No transactions in selected time period</p>
                          </TableCell>
                        </TableRow>
                      ) : (
                        transactions.map((tx) => (
                          <TableRow key={tx.transaction_id}>
                            <TableCell className="font-mono text-xs">{tx.transaction_id}</TableCell>
                            <TableCell>{getRelativeTime(tx.timestamp)}</TableCell>
                            <TableCell>${tx.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}</TableCell>
                            <TableCell>
                              <Badge variant={tx.is_fraud ? "destructive" : "outline"}>
                                {tx.transaction_type}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <div className={`h-3 w-3 rounded-full ${getRiskColor(tx.risk_score)}`} />
                                <span>{tx.risk_score}%</span>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Panel - Statistics */}
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Statistics</CardTitle>
                <CardDescription>Transaction overview</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-8">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Transactions (hour)</span>
                      <span className="font-medium">{stats.transaction_count.last_hour}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Transactions (day)</span>
                      <span className="font-medium">{stats.transaction_count.last_day}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Total transactions</span>
                      <span className="font-medium">{stats.transaction_count.total}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">High risk transactions</span>
                      <span className="font-medium">{stats.risk_levels.high}</span>
                    </div>
                    <Progress value={stats.risk_levels.high / stats.transaction_count.last_day * 100} className="h-2 bg-slate-200" indicatorClassName="bg-red-500" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Medium risk transactions</span>
                      <span className="font-medium">{stats.risk_levels.medium}</span>
                    </div>
                    <Progress value={stats.risk_levels.medium / stats.transaction_count.last_day * 100} className="h-2 bg-slate-200" indicatorClassName="bg-yellow-500" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Low risk transactions</span>
                      <span className="font-medium">{stats.risk_levels.low}</span>
                    </div>
                    <Progress value={stats.risk_levels.low / stats.transaction_count.last_day * 100} className="h-2 bg-slate-200" indicatorClassName="bg-green-500" />
                  </div>

                  <div className="rounded-lg border bg-slate-50 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Total fraud alerts</p>
                        <p className="text-2xl font-bold">{stats.fraud_count.total}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Today</p>
                        <p className="text-2xl font-bold">{stats.fraud_count.last_day}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  )
}