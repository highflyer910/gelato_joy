import { NextResponse } from 'next/server';

export async function POST(req) {
  try {
    const wishData = await req.json();
    
    const response = await fetch('http://localhost:8000/add_ice_cream_wish', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(wishData),
    });

    if (response.ok) {
      return NextResponse.json({ message: 'Ice cream wish added successfully' });
    } else {
      return NextResponse.json({ error: 'Failed to add ice cream wish' }, { status: 400 });
    }
  } catch (error) {
    console.error('Error adding ice cream wish:', error);
    return NextResponse.json({ error: 'An error occurred' }, { status: 500 });
  }
}