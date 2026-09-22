module thermodynamic

  PUBLIC :: w_to_a
  PUBLIC :: w_to_da
  PUBLIC :: get_I3
  
  ! The following subroutine translates the frequency in
  ! the a distance
  ! Note w must be in Ha units and T in K, then a is in Bohr.
contains
  subroutine w_to_a(w,T, a, n)
    double precision, intent(in) :: T
    double precision, dimension(n), intent(in) :: w
    double precision, dimension(n), intent(out) :: a
    integer, intent(in) :: n
    
    a = 0.0d0
    if (T .eq. 0.0D0) then
       a(:) = dsqrt(1.0d0 / (2.0d0 * w(:)))
    else
       a(:) = dsqrt((1.0d0 / dtanH(0.5d0 * w * &
            315774.65221921849D0 / T )) / &
            (2.0d0 * w(:)) )
    end if
  end subroutine w_to_a

  subroutine w_to_daq(w,T, da, iq, n)
    double precision, intent(in) :: T
    double precision, dimension(iq, n), intent(in) :: w
    double precision, dimension(iq, n), intent(out) :: da


    integer :: n, iq
    double precision, dimension(iq, n) :: a, b
    double precision :: beta


    if (T .eq. 0.0D0) then
       da = - dsqrt(1.0d0 / (8.0d0 *  w**3.0d0))
    else
       !    result_w_to_da = - dsqrt((1.0d0 / dtanH(0.5d0 * w * &
       !                      315774.65221921849D0 / T )) / &
       !                     (8.0d0 * m * w**3.0d0) )     * &
       !                     (1.0d0 +  (w * 315774.65221921849D0 / T) * &
       !                     (1.0d0 / dsinH(0.5d0 * w * &
       !                      315774.65221921849D0 / T )))
       beta =  315774.65221921849D0 / T
       a = w * beta + dsinH(w * beta)
       b = dsqrt(1.0d0 / (32.0d0 *  (w**3.0d0) * &
            (dsinH(0.5d0 * w * beta)**3.0) *  dcosH(0.5d0 * w * beta)))  
       da = - a * b
    end if
  end subroutine w_to_daq

  ! Computes da/dw starting from the w frequencies
  ! w unit <= Ha
  ! T unit <= K
  ! a unit => Bohr
  subroutine w_to_da(w,T, da, n)
    double precision, intent(in) :: T
    double precision, dimension(n), intent(in) :: w
    double precision, dimension(n), intent(out) :: da


    integer, intent(in) :: n
    double precision, dimension(n) :: a, b
    double precision :: beta


    if (T .eq. 0.0D0) then
       da = - dsqrt(1.0d0 / (8.0d0 *  w**3.0d0))
    else
       !    result_w_to_da = - dsqrt((1.0d0 / dtanH(0.5d0 * w * &
       !                      315774.65221921849D0 / T )) / &
       !                     (8.0d0 * m * w**3.0d0) )     * &
       !                     (1.0d0 +  (w * 315774.65221921849D0 / T) * &
       !                     (1.0d0 / dsinH(0.5d0 * w * &
       !                      315774.65221921849D0 / T )))
       beta =  315774.65221921849D0 / T
       a = w * beta + dsinH(w * beta)
       b = dsqrt(1.0d0 / (32.0d0 *  (w**3.0d0) * &
            (dsinH(0.5d0 * w * beta)**3.0) *  dcosH(0.5d0 * w * beta)))  
       da = - a * b
    end if
  end subroutine w_to_da

  
  ! This function calculates the value of the derivative of the
  ! total free energy of the harmonic oscillator minus the potential
  ! of the harmonic oscillator with respect to the arbitrary frequency:
  !
  ! d [ F_0 - 1/2 m W^2 <u^2>_0 ] / dW   
  ! 
  ! The frequency needs to
  ! be given in Ha, the temperature in K and the mass in Ha atomic
  ! units.
  
  function dW_f0_u0(w,T) result(result_dW_f0_u0)

    double precision, intent(in) :: w, T
    double precision :: result_dW_f0_u0

    if (T .eq. 0.0d0) then
       result_dW_f0_u0 = 0.25d0
    else 
       result_dW_f0_u0 = 0.25d0 * (2.0d0 * nb(w,T) + 1.0d0 +    & 
            2.0d0 * (315774.65221921849D0 / T) * w * &
            dexp(w * 315774.65221921849D0 / T) *      &
            nb(w,T)**2.0d0)
    end if
  end function dW_f0_u0

  
  ! This function calculates the Bose-Einstein distribution function for
  ! temperature T given in K and frequency given in Ha
  function nb(w,T) result(result_nb)
    double precision, intent(in) :: w, T
    double precision :: result_nb
    !  double precision :: T          !MODIFIED
    !  double complex   :: w          !MODIFIED
    !  double complex   :: result_nb  !MODIFIED

    if (T .eq. 0.0D0) then
       result_nb = 0.0D0
    else
       result_nb = 1.0D0 / (dexp(w * 315774.65221921849D0 / T )  - 1.0D0)
    end if

  end function nb

    subroutine get_I3(wq, refq3, transq, T, I3, &
    iq,n_mode,nrefq3,dimq3)

        implicit none

        double precision, dimension(iq, n_mode), intent(in) :: wq
        integer, dimension(nrefq3,dimq3,3), intent(in) :: refq3
        logical, dimension(iq, n_mode), intent(in) :: transq

        double precision, dimension(nrefq3, n_mode, n_mode, n_mode), intent(out) :: I3
        double precision, intent(in) :: T
        
        double precision, dimension(iq, n_mode) :: nbq
        double precision, parameter :: HARTREE_TO_K = 315774.65221921849d0

        double precision :: i31, i32, i33, i34, beta, n1, n2, n3, w1, w2, w3
        integer :: iq, n_mode, nrefq3, dimq3, q1, q2, q3, mu1, mu2, mu3, rq3
        
        I3 = 0.0d0

        if (T/=0) then
            call get_nbq(wq, T, nbq, iq, n_mode)
            beta = HARTREE_TO_K/T
            do rq3 = 1, nrefq3
                q1 = refq3(rq3,1,1)+1
                q2 = refq3(rq3,1,2)+1
                q3 = refq3(rq3,1,3)+1
                do1: do mu1 = 1, n_mode
                    if (transq(q1,mu1)) cycle do1
                    do2: do mu2 = 1, n_mode
                        if (transq(q2,mu2)) cycle do2
                        do3: do mu3 = 1, n_mode
                            if (transq(q3,mu3)) cycle do3
                            n1 = nbq(q1,mu1)
                            n2 = nbq(q2,mu2)
                            n3 = nbq(q3,mu3)
                            w1 = wq(q1,mu1)
                            w2 = wq(q2,mu2)
                            w3 = wq(q3,mu3)
                            i31 = &
        ((1+n1)*(1+n2)*(1+n3)-n1*n2*n3)/(w1+w2+w3)
                            ! Scale independent check
                            if ((abs(w1-w2-w3))<0.00001d0*(w2+w3)) then
                                i32 = beta*n2*n3*(1+n1)
                            else
                                i32 = (n1*(1+n2+n3)-n2*n3)/(-w1+w2+w3)
                            end if
                            if ((abs(w2-w1-w3))<0.00001d0*(w1+w3)) then
                                i33 = beta*n1*n3*(1+n2)
                            else
                                i33 = (n2*(1+n1+n3)-n1*n3)/(-w2+w1+w3)
                            end if
                            if ((abs(w3-w2-w1))<0.00001d0*(w2+w1)) then
                                i34 = beta*n2*n1*(1+n3)
                            else
                                i34 = (n3*(1+n2+n1)-n2*n1)/(-w3+w2+w1)
                            end if
                            I3(rq3,mu1,mu2,mu3) = (i31+i32+i33+i34)/(w1*w2*w3)
                        end do do3
                    end do do2
                end do do1
            end do
        else
            ! The simplified T=0 case.
            do rq3 = 1, nrefq3
                q1 = refq3(rq3,1,1)+1
                q2 = refq3(rq3,1,2)+1
                q3 = refq3(rq3,1,3)+1
                do4: do mu1 = 1, n_mode
                    if (transq(q1,mu1)) cycle do4
                    do5: do mu2 = 1, n_mode
                        if (transq(q2,mu2)) cycle do5
                        do6: do mu3 = 1, n_mode
                            if (transq(q3,mu3)) cycle do6
                            w1 = wq(q1,mu1)
                            w2 = wq(q2,mu2)
                            w3 = wq(q3,mu3)
                            I3(rq3,mu1,mu2,mu3) = 1/(w1*w2*w3*(w1+w2+w3))
                        end do do6
                    end do do5
                end do do4
            end do
        end if
    end subroutine get_I3

    subroutine get_nbq(wq, T, nbq, &
    iq, n_mode)
        implicit none
        
        double precision, intent(in) :: T
        double precision, dimension(iq, n_mode), intent(in) :: wq
        double precision, dimension(iq, n_mode), intent(out) :: nbq

        double precision, parameter :: HARTREE_TO_K = 315774.65221921849d0
        double precision :: beta, wb
        integer :: iq, n_mode, qmu, mu

        ! Zero temperature limit
        if (T <= 0.0d0) then
            nbq(:,:) = 0.0d0
        else
            ! Precompute invariant factor outside array operation
            beta = HARTREE_TO_K / T

            ! Loop/Array operation (vectorizes cleanly with compiler flags)
            do qmu = 1, iq
                do mu = 1, n_mode
                    wb = wq(qmu,mu) * beta
                    if (wb > 700.0d0) then
                        ! Prevent double-precision exponential overflow (e^700 ~ 1e304)
                        nbq(qmu,mu) = 0.0d0
                    else if (wq(qmu,mu) <= 1.0d-12) then
                        ! Guard against zero/negative frequencies (e.g., acoustic gamma modes)
                        nbq(qmu,mu) = 0.0d0
                    else
                        nbq(qmu,mu) = 1.0d0 / (dexp(wb) - 1.0d0)
                    end if
                end do
            end do
        end if

    end subroutine get_nbq
end module thermodynamic
